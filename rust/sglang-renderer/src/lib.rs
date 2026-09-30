//! Reusable request preprocessing for SGLang.
//!
//! The core lowers chat and text completions to the token-in contract consumed
//! by SGLang. Chat templates, tokenization, and output parsing come from
//! `sglang-processor`.
//!
//! OpenAI operations and generation decoding are independent of transport.
//! HTTP adapters, the SGLang HTTP engine client, and the process runtime serve
//! them. Protocol adapters own middleware and framing; shared services own
//! request preparation, submission policy, and decoding.

mod config;
mod engine;
mod error;
mod frontend;
mod launcher;
mod openai;
mod preprocessing;
mod runtime;
mod types;

pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub(crate) use engine::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenLogprob,
};
pub use error::{
    RendererError, RendererErrorKind, ResponseError, ResponseErrorKind, UpstreamErrorCode,
};
pub use launcher::run_cli;
pub use preprocessing::{
    ChatGenerateRequest, GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams,
    GenerationOptions, PreparedChat, RendererService, SamplingParams, TextRequest, TokenIdsRequest,
};
pub(crate) use preprocessing::{GenerateRequestIdentity, TextRequestGroup};
pub use runtime::{RendererRuntimeConfig, serve};
pub use types::{OneOrMany, TokenIds};
