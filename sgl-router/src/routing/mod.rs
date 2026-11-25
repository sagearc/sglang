// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::buckets_reorg::{BucketResolver, ReadyPools};
use crate::config::Config;
use crate::discovery::{ModelId, WorkerMode};
use crate::policies::{buckets::BucketSelector, registry::PdPoolResolver, PolicyRegistry};
use crate::proxy::Proxy;
use crate::server::{
    app_context::AppContext, error::ApiError, metrics::MetricsRegistry,
    routes::worker_info::WorkerPools,
};
use crate::state::kv_events::{
    BlockSizeOracle, KvEventIndex, KvIndexMetrics, RadixTreePrefixProvider,
};
use crate::state::load_monitor::{
    engine_reported_load::EngineReportedLoadTable, router_inflight_load::RouterInflightLoadRegistry,
};
use crate::tokenizer::TokenizerRegistry;
use crate::workers::{control::WorkerControlService, WorkerRegistry};
use rand::seq::IteratorRandom;

use decode_gate::DecodeGate;
use selection::pool_error;
use std::{collections::HashMap, sync::Arc};

#[path = "../buckets_reorg.rs"]
pub mod buckets_reorg;
#[path = "../config/cli.rs"]
pub mod cli;
#[path = "../state/kv_events/mod.rs"]
pub mod kv_events;
#[path = "../workers/manager.rs"]
pub mod manager;
#[path = "../policies/mod.rs"]
pub mod policies;
#[path = "../policies_reorg/mod.rs"]
pub mod policies_reorg;
#[path = "../server/routes/workers.rs"]
pub mod worker_api;

#[path = "../workers/decode_gate.rs"]
pub mod decode_gate;
#[path = "../state/load_monitor/engine_ranking.rs"]
pub(crate) mod engine_ranking;

pub(crate) mod dispatch;
pub(crate) mod selection;

mod kv_metrics;
mod snapshot;

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

    pub(crate) fn kv_bootstrap_admit_ready(&self) -> bool {
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
    pub(crate) fn info_pools(&self, model: &ModelId) -> Result<WorkerPools, ApiError> {
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
                Ok(WorkerPools { firsts, decoders })
            }
            ChatRouting::Reorg(resolvers) => resolvers
                .get(model)
                .into_iter()
                .flat_map(|resolver| &resolver.buckets)
                .map(|bucket| bucket.ready_pools(&self.registry, model))
                .filter(|pools| !pools.firsts.is_empty())
                .map(|ReadyPools { firsts, decoders }| WorkerPools { firsts, decoders })
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

    /// Readiness conditions 3 and 4, via
    /// [`BootstrapTracker::admit_ready`](crate::state::kv_events::BootstrapTracker::admit_ready).
    /// Always true when this router holds no KV index.
    pub fn kv_bootstrap_admit_ready(&self) -> bool {
        self.routing.kv_bootstrap_admit_ready()
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
    use axum::routing::get;
    router
        .route(
            "/workers",
            get(worker_api::list_workers).post(worker_api::create_worker),
        )
        .route(
            "/workers/{*id}",
            get(worker_api::get_worker)
                .put(worker_api::replace_worker)
                .delete(worker_api::remove_worker),
        )
        .route(
            kv_events::bootstrap::SNAPSHOT_PATH,
            get(snapshot::kv_snapshot),
        )
}

#[cfg(test)]
mod readiness_tests {
    use super::*;
    use crate::server::routes::health::{
        readyz,
        tests::{add_test_worker, test_ctx},
    };
    use crate::state::kv_events::BootstrapTracker;
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn readiness_requires_resolved_model_and_both_pd_roles() {
        use crate::discovery::{WorkerId, WorkerMode, WorkerSpec};

        use WorkerMode::{Decode, Plain, Prefill};

        for (model, modes, ready) in [
            (None, vec![Plain], false),
            (Some("other"), vec![Plain], false),
            (Some("stub-model"), vec![Prefill], false),
            (Some("stub-model"), vec![Decode], false),
            (Some("stub-model"), vec![Prefill, Decode], true),
        ] {
            let ctx = test_ctx(true, false);
            for (i, mode) in modes.into_iter().enumerate() {
                ctx.routing
                    .registry
                    .add(WorkerSpec {
                        id: WorkerId(i.to_string()),
                        url: format!("http://worker-{i}:30000"),
                        mode,
                        model_ids: model.map(|m| ModelId(m.into())).into_iter().collect(),
                        bootstrap_port: (mode == Prefill).then_some(8997),
                        ..Default::default()
                    })
                    .unwrap();
            }
            assert_eq!(readyz(State(ctx)).await == StatusCode::OK, ready);
        }
    }

    #[tokio::test]
    async fn readiness_rejects_portless_prefill() {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        let ctx = test_ctx(true, false);
        for mode in [WorkerMode::Prefill, WorkerMode::Decode] {
            ctx.routing
                .registry
                .add(WorkerSpec {
                    id: WorkerId(format!("{mode:?}")),
                    url: format!("http://{mode:?}:30000"),
                    mode,
                    model_ids: vec![ModelId(ctx.config.model.id.clone())],
                    ..Default::default()
                })
                .unwrap();
        }
        assert_eq!(readyz(State(ctx)).await, StatusCode::SERVICE_UNAVAILABLE);
    }

    async fn readyz_status(ctx: Arc<AppContext>) -> StatusCode {
        crate::server::app::build_router(ctx)
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    /// A ready context with one worker whose KV index uses `tracker`.
    fn ctx_with_tracker(tracker: Arc<BootstrapTracker>) -> Arc<AppContext> {
        use crate::state::kv_events::{BlockSizeOracle, KvEventIndex};
        let mut ctx = AppContext::stub();
        ctx.mark_ready();
        add_test_worker(&ctx);
        ctx.routing.kv_index = KvEventIndex::new_with_bootstrap(
            reqwest::Client::new(),
            BlockSizeOracle::new(),
            tracker,
        )
        .snapshot_source();
        Arc::new(ctx)
    }

    /// A context whose seed gate is the ONLY thing that can hold `/readyz`
    /// down: the rank is terminal, so condition 3 is satisfied and a 503 cannot
    /// come from it.
    fn ctx_with_seed_gate(seed_required: bool, failed: bool) -> Arc<AppContext> {
        use crate::state::kv_events::bootstrap::SweepOutcome;
        use crate::state::kv_events::{BootstrapState, KvWorkerId};

        let tracker = Arc::new(BootstrapTracker::new_with_opts(
            std::time::Duration::from_secs(300),
            std::time::Duration::from_secs(120),
            seed_required,
        ));
        let rank = KvWorkerId::new("http://w1:30000".into(), 0);
        tracker.register(std::slice::from_ref(&rank));
        tracker.set(&rank, BootstrapState::Failed);
        assert!(tracker.settled(), "condition 3 must not be what gates here");
        // The verdict that terminated the rank. Failed: nine siblings were
        // there and none gave up its tree, the shape a first-wave surge pod
        // hits on a rolling update. Otherwise: they proved they hold nothing.
        let verdict = if failed {
            SweepOutcome::TimedOut
        } else {
            SweepOutcome::FleetCold
        };
        tracker.record_sweep_result(verdict, 9);
        ctx_with_tracker(tracker)
    }

    /// The gate's reason for existing: a replica that could not pull the
    /// fleet's tree must stay OUT of the Service, so a rolling update stalls
    /// with the previous generation serving instead of completing with
    /// cache-blind replicas.
    #[tokio::test]
    async fn readyz_503_when_a_required_seed_failed() {
        assert_eq!(
            readyz_status(ctx_with_seed_gate(true, true)).await,
            StatusCode::SERVICE_UNAVAILABLE,
        );
    }

    /// Same failed sweep, gate not opted in: today's behaviour is preserved
    /// exactly, so enabling the flag is the only thing that can change a
    /// fleet's rollout semantics.
    #[tokio::test]
    async fn readyz_200_on_the_same_failure_when_the_gate_is_off() {
        assert_eq!(
            readyz_status(ctx_with_seed_gate(false, true)).await,
            StatusCode::OK,
        );
    }

    /// Answering 200 once latches the gate: a sweep started by a
    /// late-discovered worker must not pull a serving replica back out of the
    /// Service; an unready replica also leaves its own EndpointSlice, so its
    /// siblings lose a bootstrap source too.
    #[tokio::test]
    async fn readyz_stays_200_after_a_later_sweep_fails() {
        use crate::state::kv_events::bootstrap::SweepOutcome;
        let ctx = ctx_with_seed_gate(true, false);
        assert_eq!(readyz_status(Arc::clone(&ctx)).await, StatusCode::OK);

        ctx.routing
            .kv_index
            .as_ref()
            .expect("index attached")
            .bootstrap()
            .record_sweep_result(SweepOutcome::TimedOut, 9);
        assert_eq!(
            readyz_status(ctx).await,
            StatusCode::OK,
            "an already-serving replica may not be un-readied",
        );
    }

    /// Condition 3 on its own: a rank still mid-bootstrap holds readiness down
    /// whether or not the seed gate is configured.
    #[tokio::test]
    async fn readyz_503_while_a_rank_is_still_bootstrapping() {
        use crate::state::kv_events::KvWorkerId;

        let tracker = Arc::new(BootstrapTracker::new(std::time::Duration::from_secs(300)));
        tracker.register(&[KvWorkerId::new("http://w1:30000".into(), 0)]);
        assert!(!tracker.settled());

        assert_eq!(
            readyz_status(ctx_with_tracker(tracker)).await,
            StatusCode::SERVICE_UNAVAILABLE,
        );
    }
}

#[cfg(test)]
mod worker_info_tests;

#[cfg(test)]
mod request_preparation_tests;
