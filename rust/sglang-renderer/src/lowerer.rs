//! Engine-free OpenAI protocol lowering.

use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};

use crate::openai::{LoweredChat, lower_chat_request, lower_completion_request};
use crate::{ChatFormatter, GenerationInput, RendererConfig, RendererError};

pub struct OpenAIRequestLowerer {
    config: RendererConfig,
    chat_formatter: Option<ChatFormatter>,
}

impl OpenAIRequestLowerer {
    pub fn new(config: RendererConfig, chat_formatter: Option<ChatFormatter>) -> Self {
        Self {
            config,
            chat_formatter,
        }
    }

    pub fn lower_chat(
        &self,
        request: &mut CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<LoweredChat, RendererError> {
        lower_chat_request(
            &self.config,
            self.chat_formatter.clone(),
            request,
            response_id,
        )
    }

    pub fn lower_completions(
        &self,
        request: &CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<GenerationInput>, RendererError> {
        lower_completion_request(&self.config, request, response_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OneOrMany, SamplingDefaults, load_chat_formatter};

    fn lowerer() -> OpenAIRequestLowerer {
        OpenAIRequestLowerer::new(
            RendererConfig {
                served_model_name: "model".into(),
                tool_call_parser: None,
                default_sampling_params: SamplingDefaults::default(),
            },
            Some(load_chat_formatter(None, None, Some("chatml")).unwrap()),
        )
    }

    #[test]
    fn chat_lowering_applies_template_and_template_stops() {
        let mut request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stop": "client-stop"
        }))
        .unwrap();

        let lowered = lowerer().lower_chat(&mut request, "chatcmpl-test").unwrap();
        let GenerationInput::Text(request) = &lowered.generation_inputs[0] else {
            panic!("Chat must lower to a text prompt");
        };

        assert!(request.text.contains("<|im_start|>user"));
        assert!(matches!(
            request.options.sampling_params.stop.as_ref(),
            Some(OneOrMany::Many(stops))
                if stops.iter().map(String::as_str).collect::<Vec<_>>()
                    == ["<|endoftext|>", "<|im_end|>", "client-stop"]
        ));
    }

    #[test]
    fn completion_lowering_preserves_text_and_token_id_prompt_states() {
        let text: CreateCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": "hello"
        }))
        .unwrap();
        let token_ids: CreateCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": [11, 12, 13]
        }))
        .unwrap();

        assert!(matches!(
            &lowerer().lower_completions(&text, "cmpl-text").unwrap()[0],
            GenerationInput::Text(request) if request.text == "hello"
        ));
        assert!(matches!(
            &lowerer()
                .lower_completions(&token_ids, "cmpl-tokens")
                .unwrap()[0],
            GenerationInput::TokenIds(request) if request.input_ids == [11, 12, 13]
        ));
    }
}
