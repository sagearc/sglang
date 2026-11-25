// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup};
use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use crate::health::circuit_breaker::CircuitBreakerConfig;
use crate::policies_reorg::power_of_two::PowerOfTwoPolicy;
use crate::policies_reorg::Policy;
use crate::routing::ChatRouting;
use crate::server::app_context::AppContext;
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
use crate::workers::worker::WireProtocol;
use axum::body::Body;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::routing::get;
use axum::Router;
use http_body_util::BodyExt;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower::ServiceExt;

/// A fake engine whose `/server_info`, `/model_info` and
/// `/health_generate` answer `status` with a body naming `name` and the
/// caller's `Authorization` header.
async fn spawn_engine(name: &'static str, status: StatusCode) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let answer = move |path: &'static str| {
        move |headers: HeaderMap| async move {
            let auth = headers
                .get(AUTHORIZATION)
                .map(|v| v.to_str().unwrap().to_string());
            (
                status,
                axum::Json(serde_json::json!({"worker": name, "path": path, "auth": auth})),
            )
        }
    };
    let app = Router::new()
        .route("/server_info", get(answer("/server_info")))
        .route("/model_info", get(answer("/model_info")))
        .route("/health_generate", get(answer("/health_generate")));
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://127.0.0.1:{port}")
}

fn ctx_with(workers: &[(&str, WorkerMode)]) -> Arc<AppContext> {
    let ctx = AppContext::stub();
    for (i, (url, mode)) in workers.iter().enumerate() {
        ctx.routing
            .registry
            .add(WorkerSpec {
                id: WorkerId(format!("w-{i}")),
                url: (*url).to_string(),
                mode: *mode,
                bootstrap_port: (*mode == WorkerMode::Prefill).then_some(8998),
                model_ids: vec![ModelId("stub-model".into())],
                ..Default::default()
            })
            .expect("worker accepted");
    }
    Arc::new(ctx)
}

async fn get_json(ctx: Arc<AppContext>, uri: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .uri(uri)
        .header(AUTHORIZATION, "Bearer k")
        .body(Body::empty())
        .unwrap();
    let res = crate::server::app::build_router(ctx)
        .oneshot(req)
        .await
        .unwrap();
    let status = res.status();
    assert_eq!(res.headers()[CONTENT_TYPE], "application/json");
    let body = res.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&body).unwrap())
}

#[tokio::test]
async fn info_paths_forward_to_a_worker_with_the_callers_auth() {
    let plain = spawn_engine("plain", StatusCode::OK).await;
    let ctx = ctx_with(&[(&plain, WorkerMode::Plain)]);
    for (uri, path) in [
        ("/server_info", "/server_info"),
        ("/get_server_info", "/server_info"),
        ("/model_info", "/model_info"),
        ("/get_model_info", "/model_info"),
        ("/health_generate", "/health_generate"),
    ] {
        let (status, body) = get_json(Arc::clone(&ctx), uri).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
        assert_eq!(body["path"], path, "{uri}");
        assert_eq!(body["auth"], "Bearer k", "{uri}");
    }
}

/// Under P/D, `/server_info` reports decode's state, and `/model_info` a prefill's.
#[tokio::test]
async fn info_paths_under_pd() {
    let prefill = spawn_engine("prefill", StatusCode::OK).await;
    let decode = spawn_engine("decode", StatusCode::OK).await;
    let ctx = ctx_with(&[
        (&decode, WorkerMode::Decode),
        (&prefill, WorkerMode::Prefill),
    ]);
    for (uri, worker) in [
        ("/server_info", "decode"),
        ("/get_server_info", "decode"),
        ("/model_info", "prefill"),
    ] {
        let (_, body) = get_json(Arc::clone(&ctx), uri).await;
        assert_eq!(body["worker"], worker, "{uri}");
    }
}

#[tokio::test]
async fn pd_health_generate_fails_when_either_side_fails() {
    let ok_prefill = spawn_engine("prefill", StatusCode::OK).await;
    let bad_prefill = spawn_engine("prefill", StatusCode::SERVICE_UNAVAILABLE).await;
    let ok_decode = spawn_engine("decode", StatusCode::OK).await;
    let bad_decode = spawn_engine("decode", StatusCode::SERVICE_UNAVAILABLE).await;
    for (prefill, decode, want_status, want_worker) in [
        (&ok_prefill, &ok_decode, StatusCode::OK, "decode"),
        (
            &bad_prefill,
            &ok_decode,
            StatusCode::SERVICE_UNAVAILABLE,
            "prefill",
        ),
        (
            &ok_prefill,
            &bad_decode,
            StatusCode::SERVICE_UNAVAILABLE,
            "decode",
        ),
    ] {
        let ctx = ctx_with(&[
            (prefill.as_str(), WorkerMode::Prefill),
            (decode.as_str(), WorkerMode::Decode),
        ]);
        let (status, body) = get_json(ctx, "/health_generate").await;
        assert_eq!(status, want_status, "{prefill} {decode}");
        assert_eq!(body["worker"], want_worker, "{prefill} {decode}");
    }
}

#[tokio::test]
async fn empty_fleet_is_a_503() {
    let (status, body) = get_json(ctx_with(&[]), "/server_info").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "no_healthy_workers");
}

/// A worker whose circuit breaker is open is skipped while a healthy one
/// remains, so a probe does not flap between the two.
#[tokio::test]
async fn worker_with_open_breaker_is_skipped() {
    let down = spawn_engine("down", StatusCode::SERVICE_UNAVAILABLE).await;
    let up = spawn_engine("up", StatusCode::OK).await;
    let ctx = AppContext::stub();
    for (id, url) in [("down", &down), ("up", &up)] {
        ctx.routing
            .registry
            .add_with_cb(
                WorkerSpec {
                    id: WorkerId(id.into()),
                    url: url.clone(),
                    model_ids: vec![ModelId("stub-model".into())],
                    ..Default::default()
                },
                Some(CircuitBreakerConfig::default()),
                WireProtocol::default(),
            )
            .expect("worker accepted");
    }
    let breaker = &ctx
        .routing
        .registry
        .get(&WorkerId("down".into()))
        .unwrap()
        .breaker;
    while breaker.allow() {
        breaker.record_failure();
    }
    let ctx = Arc::new(ctx);
    for _ in 0..32 {
        let (status, body) = get_json(Arc::clone(&ctx), "/health_generate").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["worker"], "up");
    }
}

/// Under bucket routing only bucket members answer, as chat would route;
/// a fleet with no ready bucket is a 503.
#[tokio::test]
async fn bucket_routing_answers_from_bucket_members_only() {
    let inside = spawn_engine("inside", StatusCode::OK).await;
    let outside = spawn_engine("outside", StatusCode::OK).await;
    let mut ctx = Arc::into_inner(ctx_with(&[
        (&inside, WorkerMode::Plain),
        (&outside, WorkerMode::Plain),
    ]))
    .unwrap();
    let bucket_of = |id: &str| {
        let policy: Arc<dyn Policy> =
            Arc::new(PowerOfTwoPolicy::new(EngineReportedLoadTable::new()));
        let group = EngineGroup {
            worker_ids: Some([WorkerId(id.into())].into()),
            ..EngineGroup::new(policy)
        };
        let resolver =
            BucketResolver::new(vec![Bucket::new("b", BucketGroups::Plain(group))]).unwrap();
        ChatRouting::Reorg([(ModelId("stub-model".into()), resolver)].into())
    };
    ctx.routing.chat_routing = bucket_of("w-0");
    let ctx = Arc::new(ctx);
    for _ in 0..32 {
        let (_, body) = get_json(Arc::clone(&ctx), "/server_info").await;
        assert_eq!(body["worker"], "inside");
    }

    let mut ctx = Arc::into_inner(ctx_with(&[(&outside, WorkerMode::Plain)])).unwrap();
    ctx.routing.chat_routing = bucket_of("absent");
    let (status, body) = get_json(Arc::new(ctx), "/health_generate").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "no_healthy_workers");
}

#[tokio::test]
async fn unreachable_worker_is_a_502() {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let url = format!("http://127.0.0.1:{port}");
    let ctx = ctx_with(&[(&url, WorkerMode::Plain)]);
    let (status, body) = get_json(ctx, "/model_info").await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"]["code"], "upstream_unreachable");
}
