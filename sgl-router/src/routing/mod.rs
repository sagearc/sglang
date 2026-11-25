// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::buckets_reorg::{BucketResolver, ReadyPools};
use crate::config::Config;
use crate::discovery::{ModelId, WorkerMode};
use crate::policies::{buckets::BucketSelector, registry::PdPoolResolver, PolicyRegistry};
use crate::proxy::Proxy;
use crate::server::{app_context::AppContext, error::ApiError, metrics::MetricsRegistry};
use crate::state::kv_events::{
    BlockSizeOracle, KvEventIndex, KvIndexMetrics, RadixTreePrefixProvider,
};
use crate::state::load_monitor::{
    engine_reported_load::EngineReportedLoadTable, router_inflight_load::RouterInflightLoadRegistry,
};
use crate::tokenizer::TokenizerRegistry;
use crate::workers::{control::WorkerControlService, DecodeGate, WorkerRegistry};
use rand::seq::IteratorRandom;

use selection::pool_error;
use std::{collections::HashMap, sync::Arc};

pub(crate) mod dispatch;
pub(crate) mod selection;
mod startup;
pub use startup::{start, WorkerTasks};

/// Routing implementation used by the standard chat-completions endpoint.
/// `--chat-routing reorg` builds resolvers with the selected reorg policy.
#[derive(Debug, Default)]
pub enum ChatRouting {
    #[default]
    Legacy,
    Reorg(HashMap<ModelId, BucketResolver>),
}

impl ChatRouting {
    /// Whether the built routing reads request tokens; `legacy` holds the legacy policies.
    pub fn needs_request_tokens(&self, legacy: &PolicyRegistry) -> bool {
        match self {
            Self::Legacy => legacy.needs_request_tokens(),
            Self::Reorg(resolvers) => resolvers.values().any(BucketResolver::needs_request_tokens),
        }
    }
}

pub struct RoutingState {
    pub registry: Arc<WorkerRegistry>,
    pub worker_control: Option<Arc<WorkerControlService>>,
    pub policies: Arc<PolicyRegistry>,
    /// Converts static Bucket configuration into request candidate domains.
    pub bucket_selector: Arc<BucketSelector>,
    /// Select legacy policies or model-specific bucket-first routing.
    pub chat_routing: ChatRouting,
    pub decode_gate: DecodeGate,
    /// Shared Engine LoadStat table; ingress captures one immutable snapshot per request.
    pub engine_reported_load: Arc<EngineReportedLoadTable>,
    pub prefix_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
    pub radix_tree_prefix_provider: Option<RadixTreePrefixProvider>,
    /// Local KV tree for `--dp-aware` rank selection, under any policy.
    pub dp_rank_prefix_provider: Option<RadixTreePrefixProvider>,
    pub block_size_oracle: Arc<BlockSizeOracle>,
    /// Read-only handles `/metrics` pulls the KV storage-tier series from on
    /// scrape. `None` when this router maintains no local tree (external
    /// Indexer), where those series would all be a structural zero — see
    /// [`crate::state::kv_events::KvEventIndex::metrics_source`].
    pub kv_metrics: Option<KvIndexMetrics>,
    /// The local KV-event index; `None` in metadata-only mode, which makes
    /// `/internal/kv_snapshot` a 404. See [`KvEventIndex::snapshot_source`].
    pub kv_index: Option<Arc<KvEventIndex>>,
}

impl RoutingState {
    pub(crate) fn new(
        config: &Config,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        metrics: &Arc<MetricsRegistry>,
    ) -> Self {
        policies.attach_metrics(Arc::clone(metrics));
        let bucket_selector = Arc::new(BucketSelector::new(config.model.bucket_config.clone()));
        let decode_gate = DecodeGate::new(config.proxy.decode_dispatch_limit);
        Self {
            registry,
            worker_control: None,
            policies,
            bucket_selector,
            chat_routing: ChatRouting::Legacy,
            decode_gate,
            prefix_index: None,
            radix_tree_prefix_provider: None,
            dp_rank_prefix_provider: None,
            block_size_oracle: BlockSizeOracle::new(),
            kv_metrics: None,
            engine_reported_load: EngineReportedLoadTable::new(),
            kv_index: None,
        }
    }

    /// Readiness conditions 3 and 4, via
    /// [`BootstrapTracker::admit_ready`](crate::state::kv_events::BootstrapTracker::admit_ready).
    /// Always true when this router holds no KV index.
    fn kv_bootstrap_admit_ready(&self) -> bool {
        self.kv_index
            .as_ref()
            .is_none_or(|idx| idx.bootstrap().admit_ready())
    }
}

impl RoutingState {
    /// The pools chat routes over. Under bucket routing, those of one bucket with
    /// ready workers, so a pair never spans two buckets; with none ready the
    /// answer is `no_healthy_workers`, as `/readyz` sees it, without the per-stage
    /// detail.
    pub(crate) fn info_pools(&self, model: &ModelId) -> Result<ReadyPools, ApiError> {
        match &self.chat_routing {
            ChatRouting::Legacy => {
                let resolver = PdPoolResolver::new(Arc::clone(&self.registry));
                let firsts = resolver
                    .prefill_candidates(model)
                    .map_err(|e| pool_error(e, model))?;
                // No decoders is not an error here: `/model_info` needs none, and
                // `select` reports their absence.
                let decoders = resolver
                    .decode_candidates(model)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|w| w.mode() == WorkerMode::Decode)
                    .collect();
                Ok(ReadyPools { firsts, decoders })
            }
            ChatRouting::Reorg(resolvers) => resolvers
                .get(model)
                .into_iter()
                .flat_map(|resolver| &resolver.buckets)
                .map(|bucket| bucket.ready_pools(&self.registry, model))
                .filter(|pools| !pools.firsts.is_empty())
                .choose(&mut rand::thread_rng())
                .ok_or_else(|| ApiError::NoHealthyWorkers {
                    model: model.0.clone(),
                }),
        }
    }

    pub(crate) fn is_ready(&self, model: &ModelId) -> bool {
        let available = match &self.chat_routing {
            ChatRouting::Legacy => PdPoolResolver::new(Arc::clone(&self.registry))
                .prefill_candidates(model)
                .is_ok_and(|workers| !workers.is_empty()),
            ChatRouting::Reorg(resolvers) => resolvers.get(model).is_some_and(|resolver| {
                resolver
                    .buckets
                    .iter()
                    .any(|bucket| bucket.has_ready_workers(&self.registry, model))
            }),
        };
        available && self.kv_bootstrap_admit_ready()
    }
}

impl AppContext {
    pub fn new(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
    ) -> Self {
        Self::with_router_inflight_load(
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            RouterInflightLoadRegistry::with_defaults(),
        )
    }

    /// Construct an [`AppContext`] with an explicit [`RouterInflightLoadRegistry`].
    /// Production wires the default (5-minute timeout, SystemTimeClock)
    /// via [`Self::new`]; tests that exercise the janitor pass a registry
    /// built with a `MockClock`.
    pub fn with_router_inflight_load(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        router_inflight_load: Arc<RouterInflightLoadRegistry>,
    ) -> Self {
        let metrics = MetricsRegistry::new();
        let routing = RoutingState::new(&config, registry, policies, &metrics);
        Self::from_parts(
            config,
            tokenizers,
            proxy,
            routing,
            router_inflight_load,
            metrics,
        )
    }
}

#[cfg(test)]
impl RoutingState {
    pub(crate) fn test_state(config: &Config, metrics: &Arc<MetricsRegistry>) -> Self {
        Self::new(
            config,
            Arc::new(WorkerRegistry::default()),
            Arc::new(PolicyRegistry::default()),
            metrics,
        )
    }
}

pub(crate) fn register_routes(
    router: axum::Router<Arc<AppContext>>,
) -> axum::Router<Arc<AppContext>> {
    use crate::server::routes::{cache, workers};
    use axum::routing::get;
    router
        .route(
            "/workers",
            get(workers::list_workers).post(workers::create_worker),
        )
        .route(
            "/workers/{*id}",
            get(workers::get_worker)
                .put(workers::replace_worker)
                .delete(workers::remove_worker),
        )
        .route(
            crate::state::kv_events::bootstrap::SNAPSHOT_PATH,
            get(cache::kv_snapshot),
        )
}
