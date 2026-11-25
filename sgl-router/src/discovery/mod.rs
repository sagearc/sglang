// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod fleet;
pub mod types;

pub use fleet::{k8s, spawn_discovery, static_urls, DISCOVERY_CHANNEL_CAP};
pub use types::*;
