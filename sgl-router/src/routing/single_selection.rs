// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;

use crate::discovery::{ModelId, WorkerId};
use crate::server::{
    app_context::AppContext, error::ApiError, routes::chat::preparation::PreparedRequest,
};

use super::{dispatch::SelectedWorkers, RoutingState};

pub(crate) struct ModelRouting<'a>(&'a RoutingState);

impl<'a> ModelRouting<'a> {
    pub(crate) fn lookup(ctx: &'a AppContext, _: &ModelId) -> Result<Self, ApiError> {
        // There is one configured model. The handler already resolved its canonical ID.
        Ok(Self(&ctx.routing))
    }

    pub(crate) fn needs_request_tokens(&self, _: &AppContext) -> bool {
        false
    }

    pub(crate) async fn select_workers(
        &self,
        _: &AppContext,
        request: &PreparedRequest,
        _: &HeaderMap,
        excluded: &[WorkerId],
    ) -> Result<SelectedWorkers, ApiError> {
        let worker = self
            .0
            .registry
            .healthy_workers_for(&request.model)
            .into_iter()
            .find(|worker| !excluded.contains(&worker.id))
            .ok_or_else(|| ApiError::NoHealthyWorkers {
                model: request.model.0.clone(),
            })?;
        Ok(SelectedWorkers {
            prefill: worker,
            decode: None,
            track_dispatch_timestamps: false,
        })
    }
}
