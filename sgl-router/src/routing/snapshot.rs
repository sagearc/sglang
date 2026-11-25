// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::state::kv_events::bootstrap::PRODUCER_CACHE_TTL;
use axum::{
    extract::{Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use serde::Deserialize;
use std::{sync::Arc, time::Duration};

/// Query string of `GET /internal/kv_snapshot`.
///
/// Every field optional, so a caller that sends none is served.
#[derive(Debug, Default, Deserialize)]
pub struct SnapshotParams {
    /// Oldest export the caller can use, in milliseconds. Named to match
    /// [`crate::state::kv_events::bootstrap::MAX_AGE_PARAM`].
    max_age_ms: Option<u64>,
    /// When true, answer with the live cursor table and no tree. Named to
    /// match [`crate::state::kv_events::bootstrap::CURSORS_ONLY_PARAM`].
    #[serde(default)]
    cursors_only: bool,
    #[serde(default)]
    include_lora: bool,
}

/// `GET /internal/kv_snapshot` — serve this replica's KV tree so a newly
/// started sibling can bootstrap from it instead of routing cache-blind.
///
/// `404 NOT_FOUND` when `RoutingState::kv_index` is `None`. Construction is
/// single-flighted and cached; see
/// [`crate::state::kv_events::KvEventIndex::peer_snapshot_body`]. The full
/// export is served gzip-encoded when `Accept-Encoding` lists `gzip`, and as
/// identity JSON otherwise.
///
/// `?max_age_ms=N` bounds how stale a cached export may be; omitted,
/// [`PRODUCER_CACHE_TTL`] applies. `?cursors_only=true` is answered from the
/// live cursor map and bypasses the export cache. `?include_lora=true` opts into
/// version-2 LoRA mappings; legacy callers receive the base-model version-1 view.
///
/// # Exposure
///
/// Unauthenticated on the main listener, like `/flush_cache`. The body is
/// block hashes and worker URLs: no prompt text and no token ids.
pub async fn kv_snapshot(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Query(params): Query<SnapshotParams>,
) -> Response {
    let Some(index) = ctx.routing.kv_index.as_ref() else {
        return json_error(
            StatusCode::NOT_FOUND,
            "cache-aware KV indexing is not enabled on this router",
        );
    };
    if params.cursors_only {
        // Read live, so there is nothing for `max_age_ms` to bound; it is
        // ignored when both are sent.
        let body = index.peer_cursors_body();
        if body.is_empty() {
            // Same shape as the full export's failure answer below: a 200
            // carrying an empty body would hand the caller JSON it cannot
            // decode. Unreachable in practice (the body is a plain struct),
            // kept so the two paths cannot drift apart.
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "cursor table could not be encoded",
            );
        }
        return json_ok(body);
    }
    let max_age = params
        .max_age_ms
        .map_or(PRODUCER_CACHE_TTL, Duration::from_millis);
    // Pre-encoded (both encodings) and cached by the producer; handing `Bytes`
    // to the body is a refcount bump, not a copy.
    let body = index
        .peer_snapshot_body_for_client(max_age, params.include_lora)
        .await;
    if body.is_empty() {
        // The encode failed. A 200 would carry a body the caller cannot
        // decode; a non-success status reads as "no snapshot here".
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "snapshot could not be encoded",
        );
    }
    let mut resp = if accepts_gzip(&headers) {
        let mut resp = json_ok(body.gzip);
        resp.headers_mut()
            .insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        resp
    } else {
        json_ok(body.identity)
    };
    resp.headers_mut()
        .insert(header::VARY, HeaderValue::from_static("accept-encoding"));
    resp
}

/// Whether `Accept-Encoding` lists the `gzip` coding without refusing it.
/// Tokens match case-insensitively; a `q=0` weight is a refusal, and other
/// weights are not ranked.
pub(crate) fn accepts_gzip(headers: &HeaderMap) -> bool {
    headers
        .get_all(header::ACCEPT_ENCODING)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .any(|coding| {
            let mut parts = coding.split(';');
            let is_gzip = parts
                .next()
                .is_some_and(|name| name.trim().eq_ignore_ascii_case("gzip"));
            let refused = parts.any(|param| {
                param
                    .trim()
                    .strip_prefix("q=")
                    .and_then(|q| q.trim().parse::<f32>().ok())
                    .is_some_and(|q| q == 0.0)
            });
            is_gzip && !refused
        })
}

/// A `200` carrying an already-encoded JSON body.
fn json_ok(body: Bytes) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        axum::body::Body::from(body),
    )
        .into_response()
}

/// A `{"error": message}` JSON response.
fn json_error(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::kv_events::bootstrap::{
        PeerSnapshot, CURSORS_ONLY_PARAM, MAX_AGE_PARAM, SNAPSHOT_PATH,
    };
    use crate::state::kv_events::{KvEventIndex, KvWorkerId};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use reqwest::Client;
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    /// An `AppContext` serving `index`, with a 64-token unigram hash config
    /// established.
    fn ctx_for(index: &Arc<KvEventIndex>) -> Arc<AppContext> {
        index.block_size_oracle().try_set(64).unwrap();
        index.block_size_oracle().set_bigram(false);
        let mut ctx = AppContext::stub();
        ctx.routing.kv_index = index.snapshot_source();
        Arc::new(ctx)
    }

    /// One block on a carrier rank, plus a cursor for a witness rank that
    /// carries none, so the cursors-only and full cursor tables differ.
    fn ctx_with_seeded_index() -> Arc<AppContext> {
        let index = KvEventIndex::new();
        index.seed_stored_block_for_test(&KvWorkerId::new("http://carrier:30000".into(), 0), 41, 7);
        index.seed_cursor_only_for_test(&KvWorkerId::new("http://witness:30000".into(), 0), 12);
        ctx_for(&index)
    }

    async fn get_snapshot(ctx: Arc<AppContext>, uri: &str) -> Response {
        crate::server::app::build_router(ctx)
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    async fn parse_snapshot(resp: Response) -> PeerSnapshot {
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&body).expect("snapshot body must be parseable")
    }

    #[tokio::test]
    async fn kv_snapshot_route_serves_parseable_json() {
        let resp = get_snapshot(ctx_with_seeded_index(), SNAPSHOT_PATH).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let snap = parse_snapshot(resp).await;
        assert_eq!(snap.format, 1);
        assert_eq!(snap.block_size, 64);
        assert!(snap.producer_ready, "a seeded tree is worth copying");
        assert_eq!(snap.nodes.len(), 1);
        // The full export lists only carriers.
        assert_eq!(snap.workers.len(), 1);
        assert_eq!(snap.workers[0].url, "http://carrier:30000");
        assert_eq!(snap.cursors, vec![(0, 41)]);
    }

    #[tokio::test]
    async fn mixed_snapshot_preserves_legacy_base_bootstrap_and_negotiates_lora() {
        use crate::state::kv_events::bootstrap::{
            fetch_snapshot, FetchAnswer, LORA_SNAPSHOT_FORMAT,
        };
        use crate::state::kv_events::BlockStored;
        let ctx = ctx_with_seeded_index();
        let index = ctx.routing.kv_index.as_ref().unwrap().clone();
        let worker = KvWorkerId::new("http://carrier:30000".into(), 0);
        index.tree().store_event(
            &worker,
            &BlockStored {
                block_hashes: vec![100],
                parent_block_hash: None,
                token_ids: vec![1, 2],
                block_size: 64,
                lora_id: None,
                lora_name: Some("adapter-a".into()),
                cache_salt: None,
                is_bigram: false,
                medium: Some("GPU".into()),
            },
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = crate::server::app::build_router(ctx.clone());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let FetchAnswer::Body(full) = fetch_snapshot(&Client::new(), &url, Some(Duration::ZERO))
            .await
            .unwrap()
        else {
            panic!("new consumer must receive a snapshot")
        };
        assert_eq!(full.format, LORA_SNAPSHOT_FORMAT);
        assert_eq!(full.nodes.len(), 2);
        assert_eq!(full.lora_blocks.len(), 1);
        // The cached legacy view keeps the same watermark and base holdings.
        let legacy = parse_snapshot(get_snapshot(ctx, SNAPSHOT_PATH).await).await;
        assert_eq!(legacy.format, 1);
        assert!(legacy.lora_blocks.is_empty());
        assert_eq!(legacy.nodes.len(), 1);
        assert_eq!(legacy.nodes[0].block_hash, 7);
        assert_eq!(legacy.workers, full.workers);
        assert_eq!(legacy.cursors, full.cursors);
        assert!(legacy.producer_ready);
        server.abort();
        index.shutdown().await;
    }

    /// No local index means 404, not an empty snapshot.
    #[tokio::test]
    async fn kv_snapshot_route_is_absent_without_a_local_tree() {
        let resp = get_snapshot(Arc::new(AppContext::stub()), SNAPSHOT_PATH).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// Asserted through `build_router`, so it covers this route's wiring: gzip
    /// for a caller that accepts it, identity otherwise, and the gzip inflates
    /// to the identity body byte for byte.
    #[tokio::test]
    async fn kv_snapshot_route_compresses_only_when_the_caller_accepts_gzip() {
        use std::io::Read;

        let ctx = ctx_with_seeded_index();
        let gzipped = crate::server::app::build_router(Arc::clone(&ctx))
            .oneshot(
                Request::builder()
                    .uri(SNAPSHOT_PATH)
                    .header(header::ACCEPT_ENCODING, "gzip")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(gzipped.status(), StatusCode::OK);
        assert_eq!(
            gzipped
                .headers()
                .get(header::CONTENT_ENCODING)
                .and_then(|v| v.to_str().ok()),
            Some("gzip"),
            "the snapshot route must compress for a caller that accepts gzip",
        );
        assert_eq!(
            gzipped
                .headers()
                .get(header::VARY)
                .and_then(|v| v.to_str().ok()),
            Some("accept-encoding"),
        );
        let compressed = gzipped.into_body().collect().await.unwrap().to_bytes();
        let mut inflated = Vec::new();
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut inflated)
            .expect("body must be valid gzip");

        let plain = get_snapshot(ctx, SNAPSHOT_PATH).await;
        assert_eq!(plain.status(), StatusCode::OK);
        assert!(
            plain.headers().get(header::CONTENT_ENCODING).is_none(),
            "a caller that never asked for gzip must get identity",
        );
        let identity = plain.into_body().collect().await.unwrap().to_bytes();
        let _: PeerSnapshot = serde_json::from_slice(&identity).unwrap();
        assert_eq!(inflated, identity, "gzip must inflate to the identity body");
    }

    #[test]
    fn accepts_gzip_matches_the_coding_token() {
        let accepts = |value: &str| {
            let mut headers = HeaderMap::new();
            headers.insert(header::ACCEPT_ENCODING, value.parse().unwrap());
            accepts_gzip(&headers)
        };
        assert!(accepts("gzip"));
        assert!(accepts("GZip"));
        assert!(accepts("br, gzip;q=0.8, deflate"));
        assert!(!accepts("identity"));
        assert!(!accepts("br, deflate"));
        assert!(!accepts("x-gzip-ish"));
        assert!(!accepts("gzip;q=0"));
        assert!(!accepts("br, gzip; q=0.0"));
        assert!(!accepts_gzip(&HeaderMap::new()));
    }

    /// `max_age_ms` is optional; every shape is served.
    #[tokio::test]
    async fn kv_snapshot_route_accepts_a_max_age_and_survives_its_absence() {
        for uri in [
            SNAPSHOT_PATH.to_string(),
            format!("{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=0"),
            format!("{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=30000"),
        ] {
            let resp = get_snapshot(ctx_with_seeded_index(), &uri).await;
            assert_eq!(resp.status(), StatusCode::OK, "{uri}");
            let snap = parse_snapshot(resp).await;
            assert_eq!(snap.nodes.len(), 1, "{uri}");
        }
    }

    /// A valueless `?cursors_only` is not a bool, so it is rejected rather
    /// than read as either answer.
    #[tokio::test]
    async fn a_valueless_cursors_only_is_rejected() {
        let resp = get_snapshot(
            ctx_with_seeded_index(),
            &format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// `?cursors_only=true`: cursors for every observed rank, and no tree.
    #[tokio::test]
    async fn kv_snapshot_route_serves_cursors_only_when_asked() {
        let resp = get_snapshot(
            ctx_with_seeded_index(),
            &format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let snap = parse_snapshot(resp).await;

        assert!(snap.nodes.is_empty(), "a cursors-only body carries no tree");
        let mut seen: Vec<(String, i64)> = snap
            .cursors
            .iter()
            .map(|&(i, seq)| (snap.workers[i as usize].url.clone(), seq))
            .collect();
        seen.sort();
        assert_eq!(
            seen,
            vec![
                ("http://carrier:30000".to_string(), 41),
                ("http://witness:30000".to_string(), 12),
            ],
            "every observed rank is reported, carrier or not",
        );
        assert!(snap.producer_ready);
    }

    /// `?cursors_only=true` reads live: it neither fills nor reads the export
    /// cache.
    #[tokio::test]
    async fn cursors_only_neither_fills_nor_reads_the_export_cache() {
        let ctx = ctx_with_seeded_index();
        let cursors = get_snapshot(
            Arc::clone(&ctx),
            &format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        )
        .await;
        assert!(parse_snapshot(cursors).await.nodes.is_empty());

        // A generous max_age would happily reuse a cached entry, so a full
        // export right after the probe proves the probe left none behind.
        let full = get_snapshot(
            Arc::clone(&ctx),
            &format!("{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=600000"),
        )
        .await;
        assert_eq!(parse_snapshot(full).await.nodes.len(), 1);

        // The export cache is now warm. A rank first seen after it was filled
        // must still appear in the next probe, which it cannot if the probe
        // answered from the cached export.
        ctx.routing
            .kv_index
            .as_ref()
            .unwrap()
            .seed_cursor_only_for_test(&KvWorkerId::new("http://late:30000".into(), 0), 5);
        let probe = parse_snapshot(
            get_snapshot(ctx, &format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true")).await,
        )
        .await;
        assert!(
            probe
                .cursors
                .iter()
                .any(|&(i, seq)| probe.workers[i as usize].url == "http://late:30000" && seq == 5),
            "a cursors-only probe must read the live map, not the cached export",
        );
    }

    /// A half-published hash config yields `producer_ready: false`.
    #[tokio::test]
    async fn a_half_published_hash_config_is_not_a_bootstrap_source() {
        let index = KvEventIndex::new();
        index.block_size_oracle().try_set(64).unwrap(); // no bigram report yet
        index.seed_stored_block_for_test(&KvWorkerId::new("http://carrier:30000".into(), 0), 1, 7);
        let mut ctx = AppContext::stub();
        ctx.routing.kv_index = index.snapshot_source();

        let snap = parse_snapshot(get_snapshot(Arc::new(ctx), SNAPSHOT_PATH).await).await;
        assert!(
            !snap.producer_ready,
            "a half-published hash config must not be advertised as copyable",
        );
        assert!(!snap.nodes.is_empty(), "the tree itself is still reported");
    }

    /// An empty tree yields `producer_ready: false`.
    #[tokio::test]
    async fn an_empty_tree_is_not_a_bootstrap_source() {
        let ctx = ctx_for(&KvEventIndex::new());
        let snap = parse_snapshot(get_snapshot(ctx, SNAPSHOT_PATH).await).await;
        assert!(!snap.producer_ready);
        assert!(snap.nodes.is_empty());
    }
}
