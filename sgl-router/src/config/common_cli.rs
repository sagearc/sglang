// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::cli::RoutingServerArgs;
use super::{
    default_host, default_port, default_proxy_request_timeout_secs, default_shutdown_drain_secs,
    default_stale_request_timeout_secs, ConflictPolicy, LogFormat, ProxyConfig, TokenizerBackend,
};
use anyhow::{Context, Result};
use reqwest::header::HeaderValue;

#[derive(clap::Args, Debug)]
pub struct ModelArgs {
    /// Model id this router serves (the OpenAI `model` field).
    #[arg(long)]
    pub model_id: String,

    /// Extra `model` value rewritten to --model-id in the forwarded body (repeatable),
    /// preserving a LoRA suffix. /v1/models lists only --model-id.
    #[arg(long = "model-alias", value_name = "MODEL")]
    pub model_alias: Vec<String>,

    /// Local tokenizer.json or HuggingFace repo id. Defaults to --model-id; honors HF_TOKEN / HF_HOME.
    #[arg(long)]
    pub tokenizer_path: Option<String>,

    /// Skip loading a tokenizer; workers tokenize requests. Load-only policies only:
    /// cache-aware routing, prefix-cache terms or filters, and length buckets require one.
    #[arg(long, conflicts_with = "tokenizer_path")]
    pub no_tokenizer: bool,

    /// Disable generated input_ids; workers tokenize messages, while routing still renders locally.
    /// Use for worker parser/template overrides or template stop strings.
    #[arg(long)]
    pub disable_input_ids_forwarding: bool,

    /// Encode backend for router tokenization. fast uses fastokens for encoding and HF for
    /// decoding, falling back to hf when fastokens cannot load the tokenizer.
    #[arg(long, value_enum, default_value = "hf", value_name = "BACKEND")]
    pub tokenizer_backend: TokenizerBackend,

    /// L1 prefix-tokenization cache budget in MiB; 0 disables it. Reuses the tokens of a
    /// previously encoded prompt up to its deepest shared special-token boundary, so a
    /// multi-turn chat encodes only its new turns.
    #[arg(long, default_value_t = 0, value_name = "MB")]
    pub tokenizer_l1_cache_mb: usize,

    /// Same as SGLang's --default-chat-template-kwargs; must match the workers.
    #[arg(long, value_name = "JSON")]
    pub default_chat_template_kwargs: Option<String>,

    /// Fleet sampling defaults as JSON, e.g. {"temperature": 1, "top_p": 0.95}.
    /// Accepts temperature, top_p, top_k, min_p, repetition_penalty,
    /// frequency_penalty, presence_penalty, and n. Numeric values fill absent
    /// fields; {"min": LO, "max": HI} bands only constrain supplied values
    /// and require reject mode. See REFERENCE.md for domains and null handling.
    #[arg(long, value_name = "JSON")]
    pub override_sampling_params: Option<String>,

    /// How to handle sampling values that differ from configured defaults:
    /// reject returns 400 before admission; allow forwards the client value.
    /// Defaults to reject. Requires --override-sampling-params.
    #[arg(
        long,
        value_enum,
        value_name = "MODE",
        requires = "override_sampling_params"
    )]
    pub sampling_param_conflict: Option<ConflictPolicy>,
}

#[derive(clap::Args, Debug)]
pub struct ServerArgs {
    /// Address to bind the HTTP server to.
    #[arg(long, default_value_t = default_host())]
    pub host: String,

    /// Port to bind the HTTP server to.
    #[arg(long, default_value_t = default_port())]
    pub port: u16,

    /// Port for the native gRPC server (`sglang.api.v1.SglangService`, which
    /// engines serve with `--grpc-port`); requests reach engines over their gRPC port.
    #[arg(long)]
    pub grpc_port: Option<u16>,

    /// Keep serving after SIGTERM with /readyz returning 503 before stopping accepts.
    /// Leave time in the pod grace period for in-flight requests; cover readiness
    /// probe failureThreshold * periodSeconds when probe-driven. 0 disables the pause.
    #[arg(long, default_value_t = default_shutdown_drain_secs())]
    pub shutdown_drain_secs: u64,

    /// Pod terminationGracePeriodSeconds for the startup drain-budget check.
    /// Omitting this assumes 30 seconds; this flag does not change the pod spec.
    #[arg(long)]
    pub termination_grace_secs: Option<u64>,

    /// API key the router sends on its own requests to workers (`/server_info`,
    /// `/model_info`). Set it to the engines' `--api-key`.
    #[arg(long)]
    pub worker_api_key: Option<String>,

    #[command(flatten)]
    pub routing: RoutingServerArgs,

    /// Upstream timeout in seconds: the whole response of a non-streaming request,
    /// the response headers of a streaming one. SGLang's chat endpoint sends those
    /// headers with the first token, so for streaming chat this bounds time to
    /// first token, queueing included. Must be greater than zero.
    #[arg(long, default_value_t = default_proxy_request_timeout_secs())]
    pub request_timeout_secs: u64,
    /// Maximum silence between upstream stream chunks, in seconds.
    #[arg(long, default_value_t = ProxyConfig::default().stream_idle_timeout_secs)]
    pub stream_idle_timeout_secs: u64,

    /// Maximum in-flight request lifetime in seconds, including streaming responses.
    /// Expiry returns 504 `stale_request_expired` before response headers are sent;
    /// after streaming starts, it aborts the body without changing the HTTP status.
    #[arg(long, default_value_t = default_stale_request_timeout_secs())]
    pub stale_request_timeout_secs: u64,

    /// Default tracing level (overridden by `RUST_LOG`).
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Log output format.
    #[arg(long, value_enum, default_value = "text")]
    pub log_format: LogFormat,
}

/// `Bearer <key>`, marked sensitive so it stays out of `Debug` output.
pub(crate) fn bearer(key: &str, option: &str) -> Result<HeaderValue> {
    let mut value = HeaderValue::from_str(&format!("Bearer {key}"))
        .with_context(|| format!("{option} is not a valid HTTP header value"))?;
    value.set_sensitive(true);
    Ok(value)
}

pub(crate) fn model_options(
    model: &ModelArgs,
) -> Result<(
    super::SamplingOverrides,
    crate::tokenizer::chat_formatter::ChatTemplateKwargs,
)> {
    use super::sampling::parse_sampling_overrides;
    let sampling_overrides = model
        .override_sampling_params
        .as_deref()
        .map(|raw| parse_sampling_overrides(raw, model.sampling_param_conflict.unwrap_or_default()))
        .transpose()?
        .unwrap_or_default();
    let default_chat_template_kwargs = model
        .default_chat_template_kwargs
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .context("--default-chat-template-kwargs must be a JSON object")?
        .unwrap_or_default();

    Ok((sampling_overrides, default_chat_template_kwargs))
}
