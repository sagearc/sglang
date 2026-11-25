// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::discovery::WorkerId;

use super::{AffinityMode, BalancedBy, PolicyKind};

/// Reorg `--bucket-config`: plain or P/D buckets whose groups each own their
/// membership, policy and admission.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BucketsConfig {
    pub buckets: Vec<BucketSpec>,
    pub ttft_slo: SloPreference,
    pub tps_slo: SloPreference,
}

/// Set either `plain` or both `prefill` and `decode`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BucketSpec {
    pub id: String,
    pub rank: u32,
    pub min_input_tokens: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub max_context_tokens: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub plain: Option<GroupSpec>,
    pub prefill: Option<GroupSpec>,
    pub decode: Option<GroupSpec>,
}

/// Omitted fields mean every engine of the role and `--policy` (power-of-two on
/// decode); admission limits left unset take `--max-inflight-requests` / `--max-kv-usage`,
/// and affinity settings left unset take `--affinity-*`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroupSpec {
    pub worker_ids: Option<HashSet<WorkerId>>,
    /// Any listed Kubernetes Service, as namespace/name; exclusive with worker_ids.
    pub worker_services: Option<HashSet<String>>,
    pub policy: Option<PolicyKind>,
    pub admission: Option<AdmissionLimits>,
    pub affinity: Option<AffinitySpec>,
}

/// Per-group `--affinity-mode`, `--affinity-balanced-by` and `--affinity-load-*`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AffinitySpec {
    pub mode: Option<AffinityMode>,
    pub balanced_by: Option<BalancedBy>,
    pub load_factor: Option<f64>,
    pub load_gap: Option<u64>,
}

/// Per-engine caps; an unset limit is not checked and unknown metrics fail open.
/// Each admits while the engine's load is below it: counts directly, usages as
/// shares, in (0, 1], of the engine's reported capacity. Checks observe load;
/// they do not reserve capacity, and the request itself may take an engine past
/// a cap.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdmissionLimits {
    pub max_running_usage: Option<f64>,
    pub max_kv_usage: Option<f64>,
    pub max_waiting_requests: Option<u64>,
    pub max_pending_prefill_tokens: Option<u64>,
    pub max_inflight_requests: Option<u64>,
}

/// Soft preference; nonpreferred buckets remain available for fallback.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloPreference {
    #[default]
    Disabled,
    SloFirst,
    BestEffort,
}
