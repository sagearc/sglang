// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use clap::Parser;
use std::num::NonZeroU32;

use crate::config::common_cli::{bearer, model_options};
pub use crate::config::common_cli::{ModelArgs, ServerArgs};
use crate::config::{
    default_cb_cool_down, default_cb_threshold, ChatRoutingKind, CircuitBreakerConfig, Config,
    DiscoveryBackend, InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig, TokenizerConfig,
};

#[derive(Parser, Debug)]
#[command(
    name = "sgl-router",
    version,
    about = "OpenAI-compatible frontend for one SGLang worker"
)]
pub struct Cli {
    #[command(flatten)]
    pub model: ModelArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    /// URL of the SGLang worker.
    #[arg(long, value_name = "URL")]
    pub worker_urls: String,
}

#[derive(clap::Args, Debug)]
pub struct RoutingServerArgs {}

impl Cli {
    pub fn chat_routing(&self) -> ChatRoutingKind {
        ChatRoutingKind::Legacy
    }

    pub fn into_config(self) -> Result<Config> {
        let (sampling_overrides, default_chat_template_kwargs) = model_options(&self.model)?;
        let worker_auth = self
            .server
            .worker_api_key
            .as_deref()
            .map(|key| bearer(key, "--worker-api-key"))
            .transpose()?;
        let config = Config {
            server: ServerConfig {
                host: self.server.host,
                port: self.server.port,
                grpc_port: self.server.grpc_port,
                shutdown_drain_secs: self.server.shutdown_drain_secs,
                termination_grace_secs: self.server.termination_grace_secs,
                worker_auth,
                worker_management_auth: None,
            },
            observability: ObservabilityConfig {
                log_level: self.server.log_level,
                log_format: self.server.log_format,
            },
            model: ModelConfig {
                tokenizer_path: (!self.model.no_tokenizer).then(|| {
                    self.model
                        .tokenizer_path
                        .unwrap_or_else(|| self.model.model_id.clone())
                }),
                id: self.model.model_id,
                aliases: self.model.model_alias,
                disable_input_ids_forwarding: self.model.disable_input_ids_forwarding,
                tokenizer: TokenizerConfig {
                    backend: self.model.tokenizer_backend,
                    l1_cache_mb: self.model.tokenizer_l1_cache_mb,
                },
                policy: PolicyKind::RoundRobin,
                decode_policy: Default::default(),
                dp_aware: false,
                dp_rank_policy: Default::default(),
                decode_dp_rank_policy: None,
                bucket_config: None,
                reorg_buckets: None,
                reorg_admission: Default::default(),
                circuit_breaker: Some(CircuitBreakerConfig {
                    threshold: default_cb_threshold(),
                    cool_down_secs: default_cb_cool_down(),
                }),
                cache_aware: None,
                sticky: None,
                affinity: None,
                fused: None,
                eligibility: None,
                sampling_overrides,
                default_chat_template_kwargs,
            },
            discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: vec![self.worker_urls],
            }),
            proxy: ProxyConfig {
                request_timeout_secs: self.server.request_timeout_secs,
                stream_idle_timeout_secs: self.server.stream_idle_timeout_secs,
                max_attempts: NonZeroU32::MIN,
                ..Default::default()
            },
            router_inflight_load: InflightLoadConfig {
                stale_request_timeout_secs: self.server.stale_request_timeout_secs,
            },
        };
        config.validate()?;
        Ok(config)
    }
}
