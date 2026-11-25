// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! sgl-router: slim KV-aware OpenAI-compatible router for SGLang workers.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod config;
pub mod health;
pub(crate) mod off_worker;
pub(crate) mod protocol;
pub mod proxy;
pub mod server;
pub mod state;
pub mod tokenizer;
pub mod workers;

pub mod discovery;
#[cfg_attr(not(feature = "routing"), path = "routing/single.rs")]
pub mod routing;
#[cfg(feature = "routing")]
pub use routing::{buckets_reorg, policies, policies_reorg};
