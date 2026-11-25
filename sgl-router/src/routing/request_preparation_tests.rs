// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::off_worker::test_support::{slow_chat_body, slow_template_dir, time_beside_a_timer};
use crate::server::routes::chat::{prepare, Endpoint, Preprocessor};
use crate::tokenizer::TokenizerRegistry;
use axum::http::HeaderMap;
use bytes::Bytes;
use std::time::Duration;

use super::{AppContext, ModelId};
use std::sync::Arc;

/// A routable context over [`slow_template_dir`] that tokenizes chat
/// requests in `prepare`. The directory must outlive the registry load.
fn slow_render_ctx(dir: &std::path::Path) -> AppContext {
    let mut ctx = AppContext::stub();
    ctx.config.model.tokenizer_path = Some(dir.join("tokenizer.json").to_str().unwrap().into());
    // A token-aware route, so `prepare` renders and tokenizes the chat.
    ctx.config.model.dp_aware = true;
    ctx.preprocessor = Preprocessor::new(
        &ctx.config.model,
        Arc::new(TokenizerRegistry::load_from_config(&ctx.config).unwrap()),
        Arc::clone(&ctx.metrics),
    );
    ctx.routing.policies.insert(
        ModelId("stub-model".into()),
        Arc::new(super::policies::round_robin::RoundRobinPolicy::new()),
    );
    ctx
}

/// With ONE worker, a slow chat render in `prepare` must not keep another
/// task's timer from firing.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn a_slow_chat_render_leaves_the_runtime_serving() {
    let dir = slow_template_dir();
    let ctx = Arc::new(slow_render_ctx(dir.path()));
    let body = slow_chat_body(
        |b| {
            assert!(ctx
                .preprocessor
                .tokenizers()
                .encode_chat("stub-model", b)
                .is_some())
        },
        Duration::from_millis(400),
    );
    let body = Bytes::from(body.to_string());
    let (tx, rx) = tokio::sync::oneshot::channel();
    let (took, ticked_after) = time_beside_a_timer(move || async move {
        let tokens = prepare(&ctx, Endpoint::Chat, &HeaderMap::new(), body)
            .await
            .map(|(request, _)| request.tokens.is_some())
            .map_err(|e| format!("{e:?}"));
        tx.send(tokens).unwrap();
    })
    .await;
    assert_eq!(
        rx.await.unwrap(),
        Ok(true),
        "prepare must have rendered and tokenized the chat"
    );
    assert!(
        ticked_after < took / 2,
        "timer task waited {ticked_after:?} while prepare took {took:?}"
    );
}
