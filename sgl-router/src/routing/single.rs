// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use std::{sync::Arc, time::Duration};
use tokio::task::JoinHandle;

use crate::{
    config::{ChatRoutingKind, Config, DiscoveryBackend},
    discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec},
    health::circuit_breaker::CircuitBreakerConfig,
    proxy::Proxy,
    server::{app_context::AppContext, metrics::MetricsRegistry},
    state::load_monitor::router_inflight_load::RouterInflightLoadRegistry,
    tokenizer::TokenizerRegistry,
    workers::{
        introspect::{dials_cleartext, resolve_protocol, worker_client, SERVER_INFO_TIMEOUT},
        WorkerIntrospector, WorkerRegistry,
    },
};

#[path = "../config/single_cli.rs"]
pub mod cli;
#[path = "../state/kv_events/shared.rs"]
pub mod kv_events;

pub struct RoutingState {
    pub registry: Arc<WorkerRegistry>,
}

pub struct WorkerTasks(JoinHandle<()>);

impl WorkerTasks {
    pub async fn shutdown(self) {
        self.0.abort();
        let _ = self.0.await;
    }
}

pub async fn start(
    config: &Config,
    _: ChatRoutingKind,
    tokenizers: Arc<TokenizerRegistry>,
    inflight: Arc<RouterInflightLoadRegistry>,
) -> Result<(Arc<AppContext>, WorkerTasks)> {
    let DiscoveryBackend::StaticUrls(static_urls) = &config.discovery else {
        anyhow::bail!("the single-engine build requires one --worker-urls URL");
    };
    let [url] = static_urls.urls.as_slice() else {
        anyhow::bail!("the single-engine build requires one --worker-urls URL");
    };
    let proxy = Proxy::new(Duration::from_secs(config.proxy.request_timeout_secs))
        .context("build proxy client")?
        .with_stream_idle_timeout(Duration::from_secs(config.proxy.stream_idle_timeout_secs));
    let registry = Arc::new(WorkerRegistry::default());
    let app = AppContext::from_parts(
        config.clone(),
        tokenizers,
        Arc::new(proxy),
        RoutingState {
            registry: Arc::clone(&registry),
        },
        inflight,
        MetricsRegistry::new(),
    );
    let worker = tokio::spawn(initialize_worker(config.clone(), url.clone(), registry));
    Ok((Arc::new(app), WorkerTasks(worker)))
}

async fn initialize_worker(config: Config, url: String, registry: Arc<WorkerRegistry>) {
    let model = ModelId(config.model.id);
    let introspector = WorkerIntrospector::with_client(worker_client(
        SERVER_INFO_TIMEOUT,
        config.server.worker_auth,
    ));
    loop {
        let mut info = introspector.fetch(&url).await;
        if info.accepts_plain_requests == Some(false) {
            tracing::error!(worker_url = %url, "the single-engine build requires a plain worker");
            return;
        }
        if info.accepts_plain_requests == Some(true)
            && info.served_model_name.as_deref() == Some(&model.0)
        {
            let protocol = resolve_protocol(info.enable_http2, dials_cleartext(&url));
            let profile = info.take_engine_profile(protocol);
            let spec = WorkerSpec {
                id: WorkerId(url.clone()),
                url,
                mode: WorkerMode::Plain,
                model_ids: vec![model],
                ..Default::default()
            };
            let breaker = config
                .model
                .circuit_breaker
                .map(|breaker| CircuitBreakerConfig {
                    threshold: breaker.threshold,
                    cool_down: Duration::from_secs(breaker.cool_down_secs),
                });
            if let Err(error) = registry.add_with_cb(spec, breaker, profile) {
                tracing::error!(%error, "could not register the single engine");
            }
            return;
        }
        if let Some(reported) = info.served_model_name.filter(|name| name != &model.0) {
            tracing::warn!(worker_url = %url, reported, expected = %model.0, "worker serves a different model");
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

impl RoutingState {
    pub(crate) fn is_ready(&self, model: &ModelId) -> bool {
        !self.registry.workers_for(model).is_empty()
    }

    pub(crate) fn info_pools(
        &self,
        model: &ModelId,
    ) -> Result<crate::server::routes::worker_info::WorkerPools, crate::server::error::ApiError>
    {
        let firsts = self.registry.healthy_workers_for(model);
        if firsts.is_empty() {
            return Err(crate::server::error::ApiError::NoHealthyWorkers {
                model: model.0.clone(),
            });
        }
        Ok(crate::server::routes::worker_info::WorkerPools {
            firsts,
            decoders: Vec::new(),
        })
    }
}

pub(crate) mod dispatch {
    pub(crate) use crate::server::routes::chat::forward::{
        dispatch_plain as dispatch, SelectedWorkers,
    };
}
#[path = "single_selection.rs"]
pub(crate) mod selection;

#[cfg(test)]
impl RoutingState {
    pub(crate) fn test_state(_: &Config, _: &Arc<MetricsRegistry>) -> Self {
        Self {
            registry: Arc::new(WorkerRegistry::default()),
        }
    }
}

pub(crate) fn register_routes(
    router: axum::Router<Arc<AppContext>>,
) -> axum::Router<Arc<AppContext>> {
    router
}

impl RoutingState {
    pub(crate) fn append_kv_metrics(&self, _: bool, _: &mut String) {}
}

#[cfg(test)]
#[path = "single_tests.rs"]
mod tests;
