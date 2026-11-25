// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Engine endpoints answered by one worker: `/server_info`, `/model_info`
//! (and their `/get_` aliases) and `/health_generate`, as SGLang engines
//! serve them. The worker is a healthy one that chat can route to (under P/D
//! a prefill with a decode peer, or for `/server_info` that decode peer),
//! picked at random. The worker's status and body pass through; only a
//! transport failure becomes a router error.

use crate::discovery::{ModelId, WorkerMode};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::workers::worker::Worker;
use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use rand::seq::SliceRandom;
use std::sync::Arc;

/// Under P/D a decode worker answers: only decode reports decoding state, such
/// as the speculative accept length in `internal_states`.
pub async fn server_info(State(ctx): State<Arc<AppContext>>, headers: HeaderMap) -> Response {
    let worker = match select(&ctx) {
        Ok((first, decode)) => decode.unwrap_or(first),
        Err(e) => return e.into_response(),
    };
    forward(&ctx, &headers, &worker, "/server_info")
        .await
        .unwrap_or_else(IntoResponse::into_response)
}

pub async fn model_info(State(ctx): State<Arc<AppContext>>, headers: HeaderMap) -> Response {
    forward_to_first_stage(&ctx, &headers, "/model_info").await
}

/// Runs a generation on one plain worker, or on one prefill worker and one of
/// its decode peers at once, and answers with the prefill's failure if any,
/// else with the decode's response.
pub async fn health_generate(State(ctx): State<Arc<AppContext>>, headers: HeaderMap) -> Response {
    const PATH: &str = "/health_generate";
    let (first, decode) = match select(&ctx) {
        Ok(selected) => selected,
        Err(e) => return e.into_response(),
    };
    let Some(decode) = decode else {
        return forward(&ctx, &headers, &first, PATH)
            .await
            .unwrap_or_else(IntoResponse::into_response);
    };
    let (prefill, decode) = tokio::join!(
        forward(&ctx, &headers, &first, PATH),
        forward(&ctx, &headers, &decode, PATH),
    );
    match (prefill, decode) {
        (Err(e), _) => e.into_response(),
        (Ok(prefill), _) if !prefill.status().is_success() => prefill,
        (Ok(_), decode) => decode.unwrap_or_else(IntoResponse::into_response),
    }
}

/// Forwards to a plain worker, or a prefill worker under P/D.
async fn forward_to_first_stage(ctx: &AppContext, headers: &HeaderMap, path: &str) -> Response {
    let model = ModelId(ctx.config.model.id.clone());
    match ctx.routing.info_pools(&model) {
        Ok(pools) => forward(ctx, headers, &pick(&pools.firsts), path)
            .await
            .unwrap_or_else(IntoResponse::into_response),
        Err(e) => e.into_response(),
    }
}

/// One plain worker, or one prefill worker and one of its decode peers.
fn select(ctx: &AppContext) -> Result<(Arc<Worker>, Option<Arc<Worker>>), ApiError> {
    let model = ModelId(ctx.config.model.id.clone());
    let WorkerPools { firsts, decoders } = ctx.routing.info_pools(&model)?;
    let first = pick(&firsts);
    if first.mode() != WorkerMode::Prefill {
        return Ok((first, None));
    }
    let peers: Vec<_> = decoders
        .into_iter()
        .filter(|d| d.version_group() == first.version_group())
        .collect();
    if peers.is_empty() {
        return Err(ApiError::NoDecodeWorkersAvailable { model: model.0 });
    }
    Ok((first, Some(pick(&peers))))
}

pub(crate) struct WorkerPools {
    pub firsts: Vec<Arc<Worker>>,
    pub decoders: Vec<Arc<Worker>>,
}

fn pick(candidates: &[Arc<Worker>]) -> Arc<Worker> {
    Arc::clone(
        candidates
            .choose(&mut rand::thread_rng())
            .expect("callers pass a non-empty pool"),
    )
}

async fn forward(
    ctx: &AppContext,
    headers: &HeaderMap,
    worker: &Worker,
    path: &str,
) -> Result<Response, ApiError> {
    let url = reqwest::Url::parse(&worker.url)
        .and_then(|base| base.join(path))
        .map_err(|e| ApiError::WorkerMisconfigured {
            worker: worker.url.clone(),
            source: e.into(),
        })?;
    let mut request = ctx
        .proxy
        .admin_client()
        .get(url.clone())
        .timeout(ctx.proxy.request_timeout);
    if let Some(auth) = headers.get(AUTHORIZATION) {
        request = request.header(AUTHORIZATION, auth);
    }
    let upstream = request.send().await.map_err(|e| {
        if e.is_timeout() {
            ApiError::UpstreamTimeout {
                worker: url.clone(),
            }
        } else {
            ApiError::UpstreamUnreachable {
                worker: url.clone(),
                source: e.into(),
            }
        }
    })?;
    let status = upstream.status();
    let content_type = upstream.headers().get(CONTENT_TYPE).cloned();
    let body = upstream
        .bytes()
        .await
        .map_err(|_| ApiError::UpstreamStatus { status })?;
    let mut response = (status, body).into_response();
    if let Some(content_type) = content_type {
        response.headers_mut().insert(CONTENT_TYPE, content_type);
    }
    Ok(response)
}
