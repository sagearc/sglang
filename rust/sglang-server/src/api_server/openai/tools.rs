//! Chat tool-call parsing and OpenAI response shaping.

use dynamo_parsers::{ToolDefinition, try_tool_call_parse_aggregate_finalize};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    FinishReason as OpenAIFinishReason, FunctionCall, FunctionType, Role,
};

use crate::message::response::ChunkEvent;

pub(super) use sglang_renderer::dynamo_parser_name;

/// Build a chat-streaming delta carrying any of the optional columns.
///
/// The deprecated `function_call` field stays `None` — tool calls go through
/// the `tool_calls` array.
#[allow(deprecated)]
pub(super) fn chat_delta(
    content: Option<String>,
    role: Option<Role>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    reasoning_content: Option<String>,
) -> ChatCompletionStreamResponseDelta {
    ChatCompletionStreamResponseDelta {
        content: content.map(ChatCompletionMessageContent::Text),
        function_call: None,
        tool_calls,
        role,
        refusal: None,
        reasoning_content,
    }
}

/// Parse tool calls out of a completed (unary) generation's content.
///
/// Returns `(content, None)` when no parser is configured or no call parses —
/// the content passes through untouched. With a parser, a successful parse
/// returns the leftover non-tool text and the calls; `parallel_tool_calls`
/// false truncates the batch to the first call, mirroring Python.
pub(super) async fn parse_chat_tool_calls(
    content: String,
    parser: Option<&str>,
    tools: Option<&[ToolDefinition]>,
    parallel_tool_calls: bool,
) -> (String, Option<Vec<ChatCompletionMessageToolCall>>) {
    let Some(parser) = parser else {
        return (content, None);
    };
    let parser = dynamo_parser_name(parser);
    match try_tool_call_parse_aggregate_finalize(&content, Some(parser), tools).await {
        Ok((mut calls, normal)) if !calls.is_empty() => {
            if !parallel_tool_calls {
                calls.truncate(1);
            }
            (
                normal.unwrap_or_default(),
                Some(
                    calls
                        .into_iter()
                        .map(|call| ChatCompletionMessageToolCall {
                            id: call.id,
                            r#type: FunctionType::Function,
                            function: FunctionCall {
                                name: call.function.name,
                                arguments: call.function.arguments,
                            },
                        })
                        .collect(),
                ),
            )
        }
        _ => (content, None),
    }
}

/// Map the scheduler's finish kind onto the OpenAI wire values. Length and
/// content-filter keep their names; everything else (including a bare abort)
/// reports as `stop`, matching Python's fallback.
pub(super) fn chat_finish_reason(output: &ChunkEvent) -> Option<OpenAIFinishReason> {
    let kind = output
        .finish_reason
        .as_ref()
        .and_then(|reason| reason.kind_name());
    kind.map(|kind| match kind {
        "length" => OpenAIFinishReason::Length,
        "content_filter" => OpenAIFinishReason::ContentFilter,
        _ => OpenAIFinishReason::Stop,
    })
}

#[cfg(test)]
mod tests {
    use super::{chat_delta, chat_finish_reason, parse_chat_tool_calls};
    use crate::message::response::ChunkEvent;
    use dynamo_parsers::tool_calling::jail::{Annotated, apply_tool_calling_jail};
    use dynamo_protocols::types::CreateChatCompletionStreamResponse as StreamResponse;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
        ChatCompletionToolChoiceOption, FinishReason as OpenAIFinishReason, FunctionCallStream,
        FunctionType, Role,
    };
    use futures::{StreamExt, stream};

    fn stream_item(text: &str, finish: Option<OpenAIFinishReason>) -> Annotated<StreamResponse> {
        Annotated {
            data: Some(StreamResponse {
                id: "chatcmpl-test".into(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: chat_delta(Some(text.into()), Some(Role::Assistant), None, None),
                    finish_reason: finish,
                    logprobs: None,
                }],
                created: 1,
                model: "model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    /// A terminal chunk with no text — what the upstream path emits for an
    /// empty `Done` frame (content `None`, not an empty string).
    fn stream_done(finish: OpenAIFinishReason) -> Annotated<StreamResponse> {
        Annotated {
            data: Some(StreamResponse {
                id: "chatcmpl-test".into(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: chat_delta(None, Some(Role::Assistant), None, None),
                    finish_reason: Some(finish),
                    logprobs: None,
                }],
                created: 1,
                model: "model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    fn choice(item: &Annotated<StreamResponse>) -> &ChatChoiceStream {
        item.data.as_ref().unwrap().choices.first().unwrap()
    }

    fn delta_text(item: &Annotated<StreamResponse>) -> String {
        match choice(item).delta.content.as_ref().unwrap() {
            ChatCompletionMessageContent::Text(text) => text.clone(),
            _ => panic!("expected a text delta"),
        }
    }

    async fn apply_jail(
        items: Vec<Annotated<StreamResponse>>,
        parser: &str,
    ) -> Vec<Annotated<StreamResponse>> {
        apply_tool_calling_jail(
            Some(parser.into()),
            Some(ChatCompletionToolChoiceOption::Auto),
            None,
            false,
            stream::iter(items),
        )
        .collect()
        .await
    }

    #[tokio::test]
    async fn streaming_jail_emits_plain_text_without_buffering() {
        let items = apply_jail(
            vec![
                stream_item("Par", None),
                stream_item("is", Some(OpenAIFinishReason::Stop)),
            ],
            "llama3_json",
        )
        .await;
        assert_eq!(items.len(), 2);
        assert_eq!(delta_text(&items[0]), "Par");
        assert_eq!(choice(&items[0]).finish_reason, None);
        assert_eq!(delta_text(&items[1]), "is");
        assert_eq!(
            choice(&items[1]).finish_reason,
            Some(OpenAIFinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn streaming_jail_buffers_a_whole_call_until_done() {
        let items = apply_jail(
            vec![stream_item(
                r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
                Some(OpenAIFinishReason::Stop),
            )],
            "llama3_json",
        )
        .await;
        assert_eq!(items.len(), 1);
        let terminal = choice(&items[0]);
        assert!(matches!(
            terminal.delta.content.as_ref(),
            Some(ChatCompletionMessageContent::Text(text)) if text.is_empty()
        ));
        assert_eq!(
            terminal.delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name,
            Some("get_weather".into())
        );
        // The terminal reason is rewritten: calls were emitted.
        assert_eq!(terminal.finish_reason, Some(OpenAIFinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn streaming_jail_detects_bare_json_without_a_start_marker() {
        let items = apply_jail(
            vec![
                stream_item(r#"{"name":"get_weather","parameters":{"#, None),
                stream_item(r#""city":"Paris"}}"#, Some(OpenAIFinishReason::Stop)),
            ],
            "llama3_json",
        )
        .await;
        assert_eq!(items.len(), 1);
        let terminal = choice(&items[0]);
        assert!(matches!(
            terminal.delta.content.as_ref(),
            Some(ChatCompletionMessageContent::Text(text)) if text.is_empty()
        ));
        assert_eq!(
            terminal.delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name,
            Some("get_weather".into())
        );
        assert_eq!(terminal.finish_reason, Some(OpenAIFinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn streaming_jail_holds_only_a_split_marker() {
        let items = apply_jail(
            vec![
                stream_item("Before <|python_", None),
                stream_item(
                    r#"tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
                    Some(OpenAIFinishReason::Stop),
                ),
            ],
            "llama3_json",
        )
        .await;
        // The safe prefix streams immediately; the held marker suffix joins
        // the next chunk, which parses into a tool call.
        assert_eq!(delta_text(&items[0]), "Before ");
        let tool_call = choice(&items[1]);
        assert!(matches!(
            tool_call.delta.content.as_ref(),
            Some(ChatCompletionMessageContent::Text(text)) if text.is_empty()
        ));
        assert_eq!(
            tool_call.delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name,
            Some("get_weather".into())
        );
        assert_eq!(tool_call.finish_reason, Some(OpenAIFinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn streaming_jail_releases_an_incomplete_marker_at_done() {
        let items = apply_jail(
            vec![
                stream_item("Before <|python_", None),
                stream_done(OpenAIFinishReason::Stop),
            ],
            "llama3_json",
        )
        .await;
        let text = items
            .iter()
            .filter_map(|item| choice(item).delta.content.as_ref())
            .filter_map(|content| match content {
                ChatCompletionMessageContent::Text(text) => Some(text.clone()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, "Before <|python_");
        assert_eq!(
            choice(&items[1]).finish_reason,
            Some(OpenAIFinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn streaming_jail_emits_a_complete_tool_call_before_done() {
        let items = apply_jail(
            vec![
                stream_item(
                    r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
                    None,
                ),
                stream_done(OpenAIFinishReason::Stop),
            ],
            "llama3_json",
        )
        .await;
        let tool_position = items
            .iter()
            .position(|item| choice(item).delta.tool_calls.is_some())
            .expect("tool call chunk");
        let terminal_position = items
            .iter()
            .position(|item| choice(item).finish_reason.is_some())
            .expect("terminal chunk");
        assert!(tool_position < terminal_position);
        // Calls were emitted, so the terminal reason is rewritten.
        assert_eq!(
            choice(&items[terminal_position]).finish_reason,
            Some(OpenAIFinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn canonical_qwen_parser_name_uses_dynamo_qwen25() {
        let (content, calls) = parse_chat_tool_calls(
            r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#.into(),
            Some("qwen"),
            None,
            true,
        )
        .await;
        assert!(content.is_empty());
        assert_eq!(calls.unwrap()[0].function.name, "get_weather");
    }

    #[tokio::test]
    async fn unary_parse_without_a_parser_passes_content_through() {
        let (content, calls) =
            parse_chat_tool_calls("<|python_tag|>call".into(), None, None, true).await;
        assert_eq!(content, "<|python_tag|>call");
        assert!(calls.is_none());
    }

    #[test]
    fn chat_finish_reason_maps_scheduler_kinds() {
        let output = |finish: serde_json::Value| ChunkEvent {
            rid: "r".into(),
            text: "x".into(),
            token_ids: vec![1],
            prompt_tokens: 1,
            completion_tokens: 1,
            finish_reason: Some(serde_json::from_value(finish).unwrap()),
            ..Default::default()
        };
        assert_eq!(
            chat_finish_reason(&output(
                serde_json::json!({"type": "stop", "matched": "</s>"})
            )),
            Some(OpenAIFinishReason::Stop)
        );
        assert_eq!(
            chat_finish_reason(&output(serde_json::json!({"type": "length", "length": 8}))),
            Some(OpenAIFinishReason::Length)
        );
        assert_eq!(
            chat_finish_reason(&output(serde_json::json!({"type": "content_filter"}))),
            Some(OpenAIFinishReason::ContentFilter)
        );
        // Unknown kinds (including a bare abort) fall back to `stop`.
        assert_eq!(
            chat_finish_reason(&output(serde_json::json!({"type": "abort"}))),
            Some(OpenAIFinishReason::Stop)
        );
        assert_eq!(
            chat_finish_reason(&ChunkEvent {
                finish_reason: None,
                ..Default::default()
            }),
            None
        );
    }

    #[test]
    fn chat_delta_carries_the_optional_columns() {
        let delta = chat_delta(
            Some("hi".into()),
            Some(Role::Assistant),
            Some(vec![ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("call_1".into()),
                r#type: Some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: Some("get_weather".into()),
                    arguments: Some("{}".into()),
                }),
            }]),
            Some("thinking".into()),
        );
        assert_eq!(
            delta.content,
            Some(ChatCompletionMessageContent::Text("hi".into()))
        );
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(delta.reasoning_content, Some("thinking".into()));
        assert_eq!(
            delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name,
            Some("get_weather".into())
        );
        assert!(chat_delta(None, None, None, None).content.is_none());
    }
}
