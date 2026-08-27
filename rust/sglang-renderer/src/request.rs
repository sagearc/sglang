//! Model-facing request states produced by protocol lowering.

use crate::{SamplingParams, TokenIds};

#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    pub sampling_params: SamplingParams,
    pub stream: bool,
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: i64,
    pub token_ids_logprob: Option<TokenIds>,
    pub return_hidden_states: bool,
    pub return_text_in_logprobs: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct TextRequest {
    pub rid: String,
    pub text: String,
    pub skip_special_tokens: bool,
    pub options: GenerationOptions,
}

#[derive(Debug, Clone)]
pub struct TokenIdsRequest {
    pub rid: String,
    pub input_ids: TokenIds,
    pub options: GenerationOptions,
}

/// Protocol lowering output before tokenizer-dependent preparation.
#[derive(Debug, Clone)]
pub enum GenerationInput {
    Text(TextRequest),
    TokenIds(TokenIdsRequest),
}

impl GenerationInput {
    pub fn options(&self) -> &GenerationOptions {
        match self {
            Self::Text(request) => &request.options,
            Self::TokenIds(request) => &request.options,
        }
    }
}
