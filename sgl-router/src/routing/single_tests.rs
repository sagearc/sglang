// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};

use axum::http::StatusCode;
use axum::{
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures::{stream, StreamExt};
use serde_json::{json, Value};
use sglang_api_types::api::v1::{self as proto, sglang_service_client::SglangServiceClient};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::config::Cli;
use crate::server::{app::build_router, grpc};

use super::*;

#[allow(
    dead_code,
    reason = "the shared fixture also supports unary RPCs for proxy tests"
)]
#[path = "../../tests/support/mock_grpc_engine.rs"]
mod mock_grpc_engine;
use mock_grpc_engine::MockEngine;

// This model id attaches the built-in V4 chat renderer to the tiny tokenizer.
const MODEL: &str = "deepseek-v4-tiny";

#[test]
fn cli_requires_exactly_one_worker_and_rejects_fleet_flags() {
    let base = ["sgl-router", "--model-id", MODEL];
    let worker = ["--worker-urls", "http://localhost:30000"];
    assert!(Cli::try_parse_from(base).is_err());
    assert!(Cli::try_parse_from(base.into_iter().chain(worker)).is_ok());
    for invalid in [
        &["http://localhost:30001"][..],
        &["--worker-urls", "http://localhost:30001"],
        &["--policy", "round_robin"],
        &["--service-discovery"],
    ] {
        assert!(Cli::try_parse_from(
            base.into_iter()
                .chain(worker)
                .chain(invalid.iter().copied())
        )
        .is_err());
    }
}

async fn start_router(worker_url: &str) -> (Arc<AppContext>, WorkerTasks) {
    let cli = Cli::try_parse_from([
        "sgl-router",
        "--model-id",
        MODEL,
        "--worker-urls",
        worker_url,
        "--tokenizer-path",
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tiny_tokenizer.json"
        ),
    ])
    .unwrap();
    let routing_config = cli.chat_routing();
    let config = cli.into_config().unwrap();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&config).unwrap());
    let (app, workers) = start(
        &config,
        routing_config,
        tokenizers,
        RouterInflightLoadRegistry::with_defaults(),
    )
    .await
    .unwrap();
    app.mark_ready();
    (app, workers)
}

async fn serve_http(app: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn one_worker_serves_http_chat_and_native_grpc_generate() {
    let ids = proto::TokenIdsOrList {
        value: Some(proto::token_ids_or_list::Value::One(proto::TokenIds {
            ids: vec![4, 5],
        })),
    };
    let expected_ids = ids.clone();
    let reply = proto::GenerateStreamItem {
        item: Some(proto::generate_stream_item::Item::Frame(
            proto::GenerateResponse {
                text: "hello".into(),
                meta_info: Some(proto::GenerateMetaInfo {
                    finish_reason: Some(proto::FinishReason::stop(None)),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )),
    };
    let engine_reply = reply.clone();
    let grpc_port = MockEngine::default()
        .stream(
            "Generate",
            move |request: tonic::Request<proto::GenerateRequest>| {
                assert_eq!(request.into_inner().input_ids.as_ref(), Some(&expected_ids));
                stream::iter([Ok(engine_reply.clone())]).boxed()
            },
        )
        .start()
        .await;
    let engine_ready = Arc::new(AtomicBool::new(false));
    let engine_http = serve_http(
        Router::new()
            .route(
                "/model_info",
                get(|| async { Json(json!({"served_model_name": MODEL})) }),
            )
            .route(
                "/server_info",
                get({
                    let ready = Arc::clone(&engine_ready);
                    move || async move {
                        let status = if ready.load(Ordering::Relaxed) {
                            StatusCode::OK
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        };
                        (
                            status,
                            Json(json!({"disaggregation_mode": "null", "grpc_port": grpc_port})),
                        )
                    }
                }),
            )
            .route(
                "/v1/chat/completions",
                post(|Json(body): Json<Value>| async { Json(body) }),
            ),
    )
    .await;

    let (app, workers) = start_router(&engine_http).await;
    let router_http = serve_http(build_router(Arc::clone(&app))).await;
    let client = reqwest::Client::new();
    let readyz = || client.get(format!("{router_http}/readyz")).send();
    assert_eq!(
        readyz().await.unwrap().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    engine_ready.store(true, Ordering::Relaxed);
    tokio::time::timeout(Duration::from_secs(8), async {
        while readyz().await.unwrap().status() != StatusCode::OK {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("router becomes ready after worker metadata is available");
    for unsupported in ["/workers", "/internal/kv_snapshot"] {
        assert_eq!(
            client
                .get(format!("{router_http}{unsupported}"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
    }
    let chat = json!({"model": MODEL, "messages": [{"role": "user", "content": "hello world"}]});
    let http_body: Value = client
        .post(format!("{router_http}/v1/chat/completions"))
        .json(&chat)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    let input_ids = http_body["input_ids"]
        .as_array()
        .expect("router-rendered HTTP prompt");
    assert!(!input_ids.is_empty());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let grpc_url = format!("http://{}", listener.local_addr().unwrap());
    let cancel = CancellationToken::new();
    let server = tokio::spawn(grpc::serve(listener, app, cancel.clone().cancelled_owned()));
    let mut grpc_client = SglangServiceClient::connect(grpc_url).await.unwrap();
    let mut chunks = grpc_client
        .generate(proto::GenerateRequest {
            input_ids: Some(ids),
            stream: Some(true),
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(chunks.message().await.unwrap(), Some(reply));
    assert!(chunks.message().await.unwrap().is_none());

    cancel.cancel();
    server.await.unwrap().unwrap();
    workers.shutdown().await;
}

#[tokio::test]
async fn prefill_and_decode_workers_never_become_ready() {
    for mode in ["prefill", "decode"] {
        let url = serve_http(Router::new().route(
            "/server_info",
            get(move || async move {
                Json(json!({"served_model_name": MODEL, "disaggregation_mode": mode}))
            }),
        ))
        .await;
        let (app, workers) = start_router(&url).await;
        tokio::time::timeout(Duration::from_secs(2), async {
            while !workers.0.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("a disaggregated worker is rejected");
        assert_eq!(
            crate::server::routes::health::readyz(axum::extract::State(app)).await,
            StatusCode::SERVICE_UNAVAILABLE,
        );
        workers.shutdown().await;
    }
}
