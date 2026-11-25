// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::ModelId;
use crate::server::app_context::AppContext;
use axum::extract::State;
use axum::http::StatusCode;
use std::sync::Arc;

/// Always returns 200 — liveness probe.
pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

/// Ready after startup only when ALL hold:
/// 1. `mark_ready()` was called.
/// 2. The configured model has a usable plain or PD pool.
/// 3. Peer bootstrap has settled (`BootstrapTracker::settled`): every rank
///    resolved, or the `--kv-bootstrap-timeout-ms` deadline passed. Always
///    true without cache-aware and `--kv-peer-selector`.
/// 4. The seed gate is open (`BootstrapTracker::seed_gate_open`). Condition 3
///    settles on the deadline either way, so it cannot tell "seeded" from
///    "gave up"; with `--kv-bootstrap-seed-required` this holds a failed seed
///    out of the Service for up to max(3x --kv-bootstrap-timeout-ms, 60s).
///
/// Conditions 3 and 4 latch on the first 200 (`BootstrapTracker::admit_ready`),
/// so a later scale-up cannot drag a serving replica back to 503.
pub async fn readyz(State(ctx): State<Arc<AppContext>>) -> StatusCode {
    if !ctx.is_ready() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }
    let model = ModelId(ctx.config.model.id.clone());
    if ctx.routing.is_ready(&model) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn healthz_always_200() {
        let app = crate::server::app::build_router(test_ctx(false, false));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn readyz_503_when_not_ready() {
        let app = crate::server::app::build_router(test_ctx(false, true));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn readyz_503_when_ready_but_registry_empty() {
        // Regression: `/readyz` previously returned 200 the moment
        // `mark_ready()` was called, even with an empty worker
        // registry. The Service would route traffic to a pod that
        // could only return 503 no_healthy_workers.
        let app = crate::server::app::build_router(test_ctx(true, false));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "ready=true + empty registry must still be 503"
        );
    }

    #[tokio::test]
    async fn readyz_200_when_ready_and_worker_registered() {
        let app = crate::server::app::build_router(test_ctx(true, true));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    pub(crate) fn add_test_worker(ctx: &AppContext) {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        ctx.routing
            .registry
            .add(WorkerSpec {
                id: WorkerId("test-w".into()),
                url: "http://test:30000".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId(ctx.config.model.id.clone())],
                ..Default::default()
            })
            .expect("test worker accepted");
    }

    pub(crate) fn test_ctx(ready: bool, with_worker: bool) -> Arc<AppContext> {
        let ctx = AppContext::stub();
        if ready {
            ctx.mark_ready();
        }
        if with_worker {
            add_test_worker(&ctx);
        }
        Arc::new(ctx)
    }
}
