// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use sgl_kv_indexer::{GrpcPrefixIndex, PrefixIndex, PrefixIndexConfig};
use std::{sync::Arc, time::Duration};
use tokio::task::JoinHandle;

use super::{manager, ChatRouting};
use crate::{
    config::{
        CachePrefixProvider, ChatRoutingKind, Config, DiscoveryBackend, KvIndexerEndpointConfig,
        PolicyKind,
    },
    discovery::{spawn_discovery, ModelId},
    policies::{factory::build_registry as build_policy_registry, PolicyRegistry},
    policies_reorg::factory::build_resolver as build_reorg_resolver,
    proxy::Proxy,
    server::app_context::AppContext,
    state::{
        kv_events::{BlockSizeOracle, BootstrapTracker, KvEventIndex, RadixTreePrefixProvider},
        load_monitor::router_inflight_load::{JanitorHandle, RouterInflightLoadRegistry},
    },
    tokenizer::TokenizerRegistry,
    workers::{
        control::{ExternalWorkerSource, WorkerControlService},
        introspect::worker_client,
        WorkerRegistry,
    },
};

pub struct WorkerTasks {
    discovery: JoinHandle<()>,
    manager: JoinHandle<()>,
    reorg_cleanup: Option<JanitorHandle>,
}

impl WorkerTasks {
    pub async fn shutdown(self) {
        self.discovery.abort();
        self.manager.abort();
        if let Some(cleanup) = self.reorg_cleanup {
            cleanup.shutdown().await;
        }
    }
}

pub async fn start(
    config: &Config,
    routing: ChatRoutingKind,
    tokenizers: Arc<TokenizerRegistry>,
    local_inflight_requests: Arc<RouterInflightLoadRegistry>,
) -> Result<(Arc<AppContext>, WorkerTasks)> {
    // (Optional) Create a gRPC client only when routing uses an external KV indexer.
    let external_kv_indexer_client = create_external_kv_indexer_client(config)?;

    // Monitor engine-reported KV-cache events and load statistics for routing.
    let engine_state = start_engine_state_monitor(config, external_kv_indexer_client.is_some());

    // Build the policies that choose which workers receive each request.
    let (routing_policies, chat_routing, reorg_cleanup) = match routing {
        ChatRoutingKind::Legacy => (
            Arc::new(
                build_policy_registry(
                    config,
                    engine_state.tree(),
                    engine_state.block_size_oracle(),
                )
                .context("build policy registry")?,
            ),
            ChatRouting::Legacy,
            None,
        ),
        ChatRoutingKind::Reorg => {
            let (resolver, cleanup) = build_reorg_resolver(
                &config.model,
                &engine_state,
                external_kv_indexer_client.clone(),
            )
            .context("build reorg policies")?;
            (
                Arc::new(PolicyRegistry::default()),
                ChatRouting::Reorg([(ModelId(config.model.id.clone()), resolver)].into()),
                cleanup,
            )
        }
    };

    // Ask the built policies, so any term or filter that reads the prompt is covered.
    ensure!(
        config.model.tokenizer_path.is_some()
            || !chat_routing.needs_request_tokens(&routing_policies),
        "--no-tokenizer is incompatible with cache-aware routing and prefix-cache terms or filters"
    );

    // Started before worker discovery so the peer set is populated early. The
    // registry's `synced` flag separates "not delivered yet" from "no siblings".
    start_peer_watch(config, &engine_state).await;

    // Discovery feeds worker changes to the manager, which maintains this routing catalog.
    let worker_registry = Arc::new(WorkerRegistry::default());
    let (worker_events, discovery_handle) =
        spawn_discovery(config).await.context("spawn discovery")?;
    let external_source = match config.discovery {
        DiscoveryBackend::K8s(_) => ExternalWorkerSource::Kubernetes,
        DiscoveryBackend::StaticUrls(_) => ExternalWorkerSource::Static,
    };
    let (worker_control, control_rx) =
        WorkerControlService::new(Arc::clone(&worker_registry), external_source);

    // Share routing dependencies with HTTP handlers and mark startup complete.
    let mut app_context = build_app_context(
        config,
        tokenizers,
        Arc::clone(&worker_registry),
        routing_policies,
        Arc::clone(&local_inflight_requests),
        &engine_state,
        external_kv_indexer_client,
    )?;
    app_context.routing.chat_routing = chat_routing;
    app_context.routing.worker_control = Some(Arc::clone(&worker_control));
    let worker_manager_handle = tokio::spawn(manager::run_with_config_and_control(
        worker_events,
        control_rx,
        worker_control,
        Some(Arc::new(config.clone())),
        Some(Arc::clone(&engine_state)),
        Some(local_inflight_requests),
        Some(Arc::clone(&app_context.metrics)),
    ));
    let app_context = Arc::new(app_context);

    Ok((
        app_context,
        WorkerTasks {
            discovery: discovery_handle,
            manager: worker_manager_handle,
            reorg_cleanup,
        },
    ))
}

fn create_external_kv_indexer_client(config: &Config) -> Result<Option<Arc<dyn PrefixIndex>>> {
    let endpoint = config
        .model
        .cache_aware
        .as_ref()
        .filter(|cache| {
            config.model.policy == PolicyKind::CacheAware
                && cache.prefix_provider == CachePrefixProvider::Indexer
        })
        .and_then(|cache| cache.kv_indexer_endpoint.as_ref());
    endpoint
        .map(|endpoint| {
            GrpcPrefixIndex::new(prefix_index_config(endpoint))
                .map(|index| Arc::new(index) as Arc<dyn PrefixIndex>)
                .context("configure KV Indexer client")
        })
        .transpose()
}

fn prefix_index_config(indexer: &KvIndexerEndpointConfig) -> PrefixIndexConfig {
    PrefixIndexConfig {
        endpoint: indexer.url.clone(),
        query_deadline: Duration::from_millis(indexer.query_timeout_ms),
        max_inflight: indexer.query_max_inflight,
    }
}

fn start_engine_state_monitor(config: &Config, use_external_indexer: bool) -> Arc<KvEventIndex> {
    let http = worker_client(Duration::from_secs(2), config.server.worker_auth.clone());
    if use_external_indexer {
        // External indexing still needs worker hash metadata and engine load, but no local KV tree.
        return KvEventIndex::new_metadata_only_with_http_and_oracle(http, BlockSizeOracle::new());
    }
    // Only a peer selector enables bootstrap; without one the tracker is
    // pre-settled and `/readyz` never waits on it.
    let bootstrap = Arc::new(
        match (&config.model.cache_aware, config.discovery.peer_selector()) {
            (Some(cache), Some(_)) => BootstrapTracker::new_with_opts(
                Duration::from_millis(cache.bootstrap_timeout_ms),
                Duration::from_millis(cache.bootstrap_fetch_timeout_cap_ms),
                cache.bootstrap_seed_required,
            ),
            _ => BootstrapTracker::disabled(),
        },
    );
    KvEventIndex::new_with_bootstrap(http, BlockSizeOracle::new(), bootstrap)
}

/// Watch the EndpointSlices of this router's own Service to keep the peer
/// registry current.
///
/// A no-op unless a peer selector is set and this router maintains its own tree
/// (no external Indexer). The CLI already rejects that combination; this keeps
/// the invariant local.
async fn start_peer_watch(config: &Config, engine_state: &Arc<KvEventIndex>) {
    let DiscoveryBackend::K8s(k8s) = &config.discovery else {
        return;
    };
    let Some(selector) = k8s.peer_selector.as_ref() else {
        return;
    };
    if engine_state.snapshot_source().is_none() {
        return;
    }
    let family = crate::discovery::k8s::peer_address_family(&config.server.host);
    // Non-fatal: routing does not depend on peer discovery.
    if let Err(e) = crate::discovery::k8s::spawn_peer_watch(
        k8s.namespace.clone(),
        selector.clone(),
        engine_state.peers(),
        family,
        i32::from(config.server.port),
    )
    .await
    {
        // Only client construction fails here; RBAC is not checked until the
        // watch's first LIST, and a denial there is logged by the watch itself.
        tracing::error!(
            error = %e,
            "kv-bootstrap: peer watch failed to start (no usable Kubernetes client \
             config); the peer set stays unsynced",
        );
    }
}

fn build_app_context(
    config: &Config,
    tokenizers: Arc<TokenizerRegistry>,
    worker_registry: Arc<WorkerRegistry>,
    routing_policies: Arc<PolicyRegistry>,
    local_inflight_requests: Arc<RouterInflightLoadRegistry>,
    engine_state: &Arc<KvEventIndex>,
    external_kv_indexer_client: Option<Arc<dyn PrefixIndex>>,
) -> Result<AppContext> {
    let block_size_oracle = engine_state.block_size_oracle();
    let proxy = Arc::new(
        Proxy::new(Duration::from_secs(config.proxy.request_timeout_secs))
            .context("build proxy client")?
            .with_stream_idle_timeout(Duration::from_secs(config.proxy.stream_idle_timeout_secs)),
    );

    let mut app_context = AppContext::with_router_inflight_load(
        config.clone(),
        tokenizers,
        proxy,
        worker_registry,
        routing_policies,
        local_inflight_requests,
    );
    // An external indexer leaves the local tree empty.
    app_context.routing.dp_rank_prefix_provider = (config.model.dp_aware
        && external_kv_indexer_client.is_none())
    .then(|| RadixTreePrefixProvider::new(engine_state.tree(), Arc::clone(&block_size_oracle)));
    app_context.routing.prefix_index = external_kv_indexer_client;
    app_context.routing.radix_tree_prefix_provider = (config.model.policy
        == PolicyKind::CacheAware
        && config
            .model
            .cache_aware
            .as_ref()
            .is_some_and(|cache| cache.prefix_provider == CachePrefixProvider::RadixTree))
    .then(|| {
        let provider =
            RadixTreePrefixProvider::new(engine_state.tree(), Arc::clone(&block_size_oracle));
        match config
            .model
            .cache_aware
            .as_ref()
            .and_then(|cache| cache.pending_prefix_consult_below)
        {
            Some(below) => provider.with_pending_consult_below(below),
            None => provider,
        }
    });
    if let Some(cache) = config.model.cache_aware.as_ref() {
        let ttl = Duration::from_millis(cache.pending_prefix_ttl_ms);
        engine_state.tree().pending().enable(ttl);
        engine_state
            .tree()
            .pending()
            .set_max_followers(cache.pending_prefix_max_followers);
        if cache.media_affinity_ttl_ms > 0 {
            let ttl = Duration::from_millis(cache.media_affinity_ttl_ms);
            let tree = engine_state.tree();
            tree.media().enable(ttl);
            tree.media_ranks().enable(ttl);
            let ids = app_context
                .preprocessor
                .tokenizers()
                .media_token_ids()
                .to_vec();
            if ids.is_empty() {
                tracing::warn!(
                    "--cache-media-affinity-ttl-ms is set but the tokenizer declares no image \
                     placeholder token; media affinity stays inactive"
                );
            }
            tree.set_media_tokens(ids);
        }
    }
    app_context.routing.block_size_oracle = block_size_oracle;
    app_context.routing.engine_reported_load = engine_state.engine_reported_load();
    app_context.routing.kv_metrics = engine_state.metrics_source();
    app_context.routing.kv_index = engine_state.snapshot_source();
    Ok(app_context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Cli;
    use clap::Parser;

    #[test]
    fn prefix_index_config_preserves_router_limits() {
        let config = prefix_index_config(&KvIndexerEndpointConfig {
            url: "http://127.0.0.1:50051".to_string(),
            query_timeout_ms: 25,
            query_max_inflight: 17,
        });
        assert_eq!(config.endpoint, "http://127.0.0.1:50051");
        assert_eq!(config.query_deadline, Duration::from_millis(25));
        assert_eq!(config.max_inflight, 17);
    }

    #[tokio::test]
    async fn dp_aware_gets_the_local_tree_under_any_policy() {
        let config = Cli::try_parse_from([
            "sgl-router",
            "--model-id=tiny",
            "--tokenizer-path=tests/fixtures/tiny_tokenizer.json",
            "--worker-urls=http://127.0.0.1:1",
            "--policy=power_of_two",
            "--dp-aware",
        ])
        .unwrap()
        .into_config()
        .unwrap();
        let ctx = build_app_context(
            &config,
            Arc::new(TokenizerRegistry::load_from_config(&config).unwrap()),
            Default::default(),
            Default::default(),
            RouterInflightLoadRegistry::with_defaults(),
            &start_engine_state_monitor(&config, false),
            None,
        )
        .unwrap();
        assert!(ctx.routing.dp_rank_prefix_provider.is_some());
    }
}
