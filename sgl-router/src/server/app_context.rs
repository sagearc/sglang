// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;
use crate::proxy::Proxy;
use crate::routing::RoutingState;
use crate::server::inflight::InflightHttp;
use crate::server::metrics::MetricsRegistry;
use crate::server::routes::chat::Preprocessor;
use crate::state::load_monitor::router_inflight_load::RouterInflightLoadRegistry;
use crate::tokenizer::TokenizerRegistry;

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// `/readyz` readiness as a one-way door: `NOT_READY -> READY -> DRAINING`,
/// and never backwards. One atomic rather than a pair of bools so the latch is
/// the transition itself — `mark_ready`'s compare-exchange simply cannot
/// succeed from `DRAINING` — instead of an invariant stated in a doc comment
/// and enforced by nobody.
const READINESS_NOT_READY: u8 = 0;
const READINESS_READY: u8 = 1;
const READINESS_DRAINING: u8 = 2;

pub struct AppContext {
    pub config: Config,
    pub preprocessor: Preprocessor,
    pub proxy: Arc<Proxy>,
    pub routing: RoutingState,
    pub router_inflight_load: Arc<RouterInflightLoadRegistry>,
    pub metrics: Arc<MetricsRegistry>,
    pub inflight_http: Arc<InflightHttp>,
    readiness: AtomicU8,
}

impl AppContext {
    pub fn from_parts(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        routing: RoutingState,
        router_inflight_load: Arc<RouterInflightLoadRegistry>,
        metrics: Arc<MetricsRegistry>,
    ) -> Self {
        router_inflight_load.attach_metrics(Arc::clone(&metrics));
        Self {
            preprocessor: Preprocessor::new(&config.model, tokenizers, Arc::clone(&metrics)),
            config,
            proxy,
            routing,
            router_inflight_load,
            metrics,
            inflight_http: InflightHttp::new(),
            readiness: AtomicU8::new(READINESS_NOT_READY),
        }
    }

    /// Report bootstrap as finished, unless the pod has already begun draining.
    /// The `DRAINING` state is a one-way door (see [`Self::mark_not_ready`]),
    /// and a compare-exchange is what enforces it: a plain store would let any
    /// later caller — a re-initialization path, a discovery-recovery hook —
    /// flip `/readyz` back to 200 seconds before the listener closes, re-arming
    /// the rolling-update race the drain exists to close.
    pub fn mark_ready(&self) {
        // Relaxed: this flag does not synchronize other state; readers only
        // care about eventual visibility, not happens-before with surrounding
        // ops. Failure means the state was already READY or is DRAINING —
        // correct in both cases, so the result is deliberately discarded.
        let _ = self.readiness.compare_exchange(
            READINESS_NOT_READY,
            READINESS_READY,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    /// Flip `/readyz` to 503, permanently. Called at the start of the SIGTERM
    /// drain so probes and any probe-driven load balancer see this pod as
    /// not-ready while the endpoint removal (triggered by the pod's
    /// `deletionTimestamp`, not by this flip) propagates. See
    /// [`crate::server::shutdown::drain_for_termination`] for which mechanism
    /// the pause is sized for.
    ///
    /// Not the inverse of [`mark_ready`](Self::mark_ready): this transition
    /// cannot be undone, because the process it announces cannot be either.
    pub fn mark_not_ready(&self) {
        self.readiness.store(READINESS_DRAINING, Ordering::Relaxed);
    }

    /// Whether bootstrap finished — only ONE term of the `/readyz` predicate,
    /// which also requires a non-empty worker registry (see
    /// `server::routes::health::readyz`).
    pub fn is_ready(&self) -> bool {
        self.readiness.load(Ordering::Relaxed) == READINESS_READY
    }

    #[cfg(test)]
    pub fn stub() -> Self {
        let config = Config {
            server: crate::config::ServerConfig {
                host: "x".into(),
                port: 0,
                ..Default::default()
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                aliases: Vec::new(),
                id: "stub-model".into(),
                tokenizer_path: Some("stub".into()),
                disable_input_ids_forwarding: false,
                tokenizer: Default::default(),
                policy: crate::config::PolicyKind::RoundRobin,
                decode_policy: Default::default(),
                dp_aware: false,
                dp_rank_policy: Default::default(),
                decode_dp_rank_policy: None,
                bucket_config: None,
                reorg_buckets: None,
                reorg_admission: Default::default(),
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
                affinity: None,
                fused: None,
                eligibility: None,
                sampling_overrides: Default::default(),
                default_chat_template_kwargs: Default::default(),
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            router_inflight_load: crate::config::InflightLoadConfig::default(),
        };
        Self::test_context(config, Arc::new(TokenizerRegistry::default()))
    }

    #[cfg(test)]
    pub(crate) fn test_context(config: Config, tokenizers: Arc<TokenizerRegistry>) -> Self {
        let metrics = MetricsRegistry::new();
        let routing = RoutingState::test_state(&config, &metrics);
        Self::from_parts(
            config,
            tokenizers,
            Arc::new(Proxy::new(std::time::Duration::from_secs(60)).expect("test proxy")),
            routing,
            RouterInflightLoadRegistry::with_defaults(),
            metrics,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_not_ready_flips_readiness_back_off() {
        let ctx = AppContext::stub();
        // stub starts not-ready; mark_ready is the readiness on-switch.
        ctx.mark_ready();
        assert!(ctx.is_ready(), "mark_ready must report ready");
        // The SIGTERM drain path needs the inverse so /readyz can flip to 503
        // before the server stops accepting.
        ctx.mark_not_ready();
        assert!(
            !ctx.is_ready(),
            "mark_not_ready must flip readiness back off",
        );
    }

    /// The safety-critical direction. A `mark_ready` reaching a draining pod
    /// would put `/readyz` back to 200 with the listener seconds from closing,
    /// silently re-arming the rolling-update race — so the latch is tested,
    /// not just documented.
    #[test]
    fn readiness_does_not_come_back_once_draining() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        ctx.mark_not_ready();

        ctx.mark_ready();
        assert!(
            !ctx.is_ready(),
            "mark_ready must not re-ready a pod that has begun draining",
        );
    }

    /// `mark_ready` is idempotent: the compare-exchange failing because the
    /// state is already READY must not be mistaken for the draining case.
    #[test]
    fn mark_ready_is_idempotent() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        ctx.mark_ready();
        assert!(
            ctx.is_ready(),
            "a second mark_ready must keep the pod ready"
        );
    }
}
