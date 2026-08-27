//! OpenAI Chat Completions endpoint and chat-template preparation.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::post,
};
use dynamo_parsers::ToolDefinition;
use dynamo_parsers::tool_calling::jail::{Annotated, apply_tool_calling_jail};
use dynamo_protocols::types::{
    ChatChoice, ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionResponseMessage, ChatCompletionTokenLogprob, ChatCompletionToolChoiceOption,
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason as OpenAIFinishReason, Role, ServiceTier as ChatServiceTier, TopLogprobs,
};
use futures::StreamExt;
use tokio::sync::mpsc;

use super::super::guard::AbortGuard;
use super::completions::completion_usage;
use super::reasoning::{ReasoningStreamSplitter, split_reasoning_unary};
use super::tools::{chat_delta, chat_finish_reason, dynamo_parser_name, parse_chat_tool_calls};
use super::{
    AppState, collect_output, error_payload, indexed_decode_stream, openai_error,
    submit_generation, unix_seconds_u32,
};
use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;
use crate::message::response::{ChunkExtras, ResponseItem};

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CreateChatCompletionRequest>, JsonRejection>,
) -> Response {
    let mut request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text(), false);
        }
    };
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let lowered = match state.request_lowerer.lower_chat(&mut request, &response_id) {
        Ok(lowered) => lowered,
        Err(error) => {
            return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false);
        }
    };
    let generation_inputs = lowered.generation_inputs;
    let parser = lowered.tool_parser;
    let tools = lowered.tools;
    // Python gates the split on `request.separate_reasoning` (default true);
    // the Dynamo request type has no such field, so it is always on when the
    // server was launched with `--reasoning-parser`.
    let reasoning_parser = state.server_args.reasoning_parser.clone();
    let stream = request.stream.unwrap_or(false);
    let n = request.n.unwrap_or(1) as usize;
    let want_logprobs = request.logprobs.unwrap_or(false);
    let parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
    let stream_tool_choice = request.tool_choice.clone();
    let uses_tool_call_structural_tag = generation_inputs
        .first()
        .is_some_and(|request| request.options().sampling_params.structural_tag.is_some());
    let service_tier = request.service_tier;
    let created = unix_seconds_u32();
    let model = request.model;
    let include_usage = request
        .stream_options
        .is_some_and(|options| options.include_usage)
        || state.server_args.stream_response_default_include_usage;
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut submitted = Vec::with_capacity(n);

    for (index, generation_input) in generation_inputs.into_iter().enumerate() {
        let native = GenerateRequest::from(generation_input);
        let rid = native.rid.clone();
        let rx = match submit_generation(&state, native, stream, &mut guard).await {
            Ok(rx) => rx,
            Err(response) => return response,
        };
        submitted.push((index, rid, rx));
    }

    if stream {
        let event_stream = chat_event_stream(
            submitted,
            guard,
            response_id,
            model,
            created,
            want_logprobs,
            include_usage,
            parser,
            reasoning_parser,
            tools,
            stream_tool_choice,
            uses_tool_call_structural_tag,
            parallel_tool_calls,
            service_tier,
        )
        .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(event_stream).into_response()
    } else {
        unary_chat(
            submitted,
            guard,
            response_id,
            model,
            created,
            want_logprobs,
            parser,
            reasoning_parser,
            tools,
            parallel_tool_calls,
            service_tier,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn unary_chat(
    submitted: Vec<(usize, Rid, mpsc::Receiver<ResponseItem>)>,
    mut guard: AbortGuard,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    tools: Option<Vec<ToolDefinition>>,
    parallel_tool_calls: bool,
    service_tier: Option<ChatServiceTier>,
) -> Response {
    let mut choices = Vec::with_capacity(submitted.len());
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0u64;

    for (index, rid, rx) in submitted {
        let output = match collect_output(rx, &mut guard, &rid).await {
            Ok(output) => output,
            Err((status, message)) => {
                return openai_error(status, message, false);
            }
        };

        if prompt_tokens == 0 {
            prompt_tokens = output.prompt_tokens;
        }
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        let logprobs = want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
        let finish_reason = chat_finish_reason(&output);
        // Split reasoning markers out of the content first (Python splits
        // before tool-call parsing too), then parse tool calls on the clean
        // normal text.
        let (reasoning_text, text) =
            split_reasoning_unary(reasoning_parser.as_deref(), &output.text, &output.token_ids);
        let (content, tool_calls) = parse_chat_tool_calls(
            text,
            parser.as_deref(),
            tools.as_deref(),
            parallel_tool_calls,
        )
        .await;
        let finish_reason = if tool_calls.is_some() {
            Some(OpenAIFinishReason::ToolCalls)
        } else {
            finish_reason
        };
        #[allow(deprecated)]
        let message = ChatCompletionResponseMessage {
            content: (!content.is_empty()).then_some(ChatCompletionMessageContent::Text(content)),
            refusal: None,
            tool_calls,
            role: Role::Assistant,
            function_call: None,
            audio: None,
            // Python: `reasoning_text if reasoning_text else None`.
            reasoning_content: (!reasoning_text.is_empty()).then_some(reasoning_text),
        };
        choices.push(ChatChoice {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            message,
            finish_reason,
            logprobs,
        });
    }

    Json(CreateChatCompletionResponse {
        id: response_id,
        choices,
        created,
        model,
        service_tier,
        system_fingerprint: None,
        object: "chat.completion".into(),
        usage: Some(completion_usage(
            prompt_tokens,
            u32::try_from(completion_tokens).unwrap_or(u32::MAX),
        )),
    })
    .into_response()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn chat_event_stream(
    submitted: Vec<(usize, Rid, mpsc::Receiver<ResponseItem>)>,
    mut guard: AbortGuard,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    include_usage: bool,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    parallel_tool_calls: bool,
    service_tier: Option<ChatServiceTier>,
) -> impl futures::Stream<Item = String> {
    let count = submitted.len();
    let raw = async_stream::stream! {
        let count = submitted.len();
        let mut rids = Vec::with_capacity(count);
        let mut streams = Vec::with_capacity(count);
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u64;
        // One stateful reasoning splitter per choice (Python keeps a
        // `reasoning_parser_dict` per index).
        let mut reasoning_splitters: Vec<ReasoningStreamSplitter> =
            if reasoning_parser.is_some() {
                (0..count)
                    .map(|_| ReasoningStreamSplitter::new(reasoning_parser.as_deref()))
                    .collect()
            } else {
                vec![]
            };
        let reasoning_enabled = !reasoning_splitters.is_empty();

        for (index, rid, rx) in submitted {
            rids.push(rid);
            streams.push(indexed_decode_stream(index, rx));
            yield Annotated {
                data: Some(CreateChatCompletionStreamResponse {
                    id: response_id.clone(),
                    choices: vec![ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, Some(Role::Assistant), None, None),
                        finish_reason: None,
                        logprobs: None,
                    }],
                    created,
                    model: model.clone(),
                    service_tier: service_tier.clone(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".into(),
                    usage: None,
                }),
                id: None,
                event: None,
                comment: None,
                error: None,
            };
        }

        let mut events = futures::stream::select_all(streams);
        while let Some((index, item)) = events.next().await {
            let Some(item) = item else {
                yield Annotated {
                    data: None,
                    id: None,
                    event: None,
                    comment: None,
                    error: Some(error_payload(StatusCode::INTERNAL_SERVER_ERROR, "response truncated before completion").to_string()),
                };
                continue;
            };
            let output = match item {
                ResponseItem::Frame(output) => output,
                ResponseItem::Done(output) => {
                    guard.disarm(&rids[index]);
                    output
                }
                ResponseItem::Error(error) => {
                    guard.disarm(&rids[index]);
                    yield Annotated {
                        data: None,
                        id: None,
                        event: None,
                        comment: None,
                        error: Some(error_payload(StatusCode::from_u16(error.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), error.to_string()).to_string()),
                    };
                    continue;
                }
                ResponseItem::Control(_) | ResponseItem::Data(_) => continue,
            };
            if let Some((code, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                yield Annotated {
                    data: None,
                    id: None,
                    event: None,
                    comment: None,
                    error: Some(error_payload(StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), message).to_string()),
                };
                continue;
            }

            if prompt_tokens == 0 {
                prompt_tokens = output.prompt_tokens;
            }
            completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
            let finish_reason = chat_finish_reason(&output);
            // Split the step's text into (reasoning, normal) deltas when
            // `--reasoning-parser` is set. Mirrors Python's per-step emission:
            // reasoning chunk first (logprobs ride it), then the content chunk.
            let mut emitted = Vec::with_capacity(2);
            if reasoning_enabled {
                let (reasoning_text, normal_text) =
                    reasoning_splitters[index].split(&output.text, &output.token_ids);
                let mut remaining_logprobs =
                    want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
                if !reasoning_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, None, None, Some(reasoning_text)),
                        finish_reason: None,
                        logprobs: remaining_logprobs.clone(),
                    });
                    remaining_logprobs = None;
                }
                if !normal_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(Some(normal_text), None, None, None),
                        finish_reason: None,
                        logprobs: remaining_logprobs,
                    });
                }
            } else {
                emitted.push(ChatChoiceStream {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    delta: chat_delta(
                        (!output.text.is_empty()).then_some(output.text),
                        None,
                        None,
                        None,
                    ),
                    finish_reason: None,
                    logprobs: want_logprobs.then(|| chat_logprobs(output.extras.as_deref())),
                });
            };
            // Flush the choice's buffered reasoning tail before its terminal
            // frame (Python `parse_stream_end`, which skips aborts — abort
            // frames already became error chunks above). Both columns flush:
            // some parsers buffer the answer text until EOF.
            if reasoning_enabled && finish_reason.is_some() {
                let (reasoning_tail, normal_tail) = reasoning_splitters[index].finish();
                if !reasoning_tail.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, None, None, Some(reasoning_tail)),
                        finish_reason: None,
                        logprobs: None,
                    });
                }
                if !normal_tail.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(Some(normal_tail), None, None, None),
                        finish_reason: None,
                        logprobs: None,
                    });
                }
            }
            // The finish reason rides the last emitted chunk (the wire format
            // the equivalence tests pin); a step whose text was entirely
            // buffered inside the parser still gets a finish-only frame.
            match emitted.last_mut() {
                Some(last) => last.finish_reason = finish_reason,
                None => emitted.push(ChatChoiceStream {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    delta: chat_delta(None, None, None, None),
                    finish_reason,
                    logprobs: None,
                }),
            }
            for choice in emitted {
                yield Annotated {
                    data: Some(CreateChatCompletionStreamResponse {
                        id: response_id.clone(),
                        choices: vec![choice],
                        created,
                        model: model.clone(),
                        service_tier: service_tier.clone(),
                        system_fingerprint: None,
                        object: "chat.completion.chunk".into(),
                        usage: None,
                    }),
                    id: None,
                    event: None,
                    comment: None,
                    error: None,
                };
            }
        }

        if include_usage {
            yield Annotated {
                data: Some(CreateChatCompletionStreamResponse {
                    id: response_id,
                    choices: vec![],
                    created,
                    model,
                    service_tier,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".into(),
                    usage: Some(completion_usage(
                        prompt_tokens,
                        u32::try_from(completion_tokens).unwrap_or(u32::MAX),
                    )),
                }),
                id: None,
                event: None,
                comment: None,
                error: None,
            };
        }
    };

    let parsed: std::pin::Pin<
        Box<dyn futures::Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send>,
    > = if let Some(parser) = parser {
        Box::pin(apply_tool_calling_jail(
            Some(dynamo_parser_name(&parser).to_owned()),
            tool_choice,
            tools,
            uses_tool_call_structural_tag,
            raw,
        ))
    } else {
        Box::pin(raw)
    };

    async_stream::stream! {
        let mut tool_calls_seen = vec![false; count];
        futures::pin_mut!(parsed);
        while let Some(mut item) = parsed.next().await {
            if let Some(response) = item.data.as_mut() {
                if !parallel_tool_calls {
                    for choice in &mut response.choices {
                        let index = choice.index as usize;
                        if let Some(calls) = choice.delta.tool_calls.as_mut() {
                            if tool_calls_seen.get(index).copied().unwrap_or(false) {
                                calls.clear();
                            } else {
                                calls.truncate(1);
                                if !calls.is_empty()
                                    && let Some(seen) = tool_calls_seen.get_mut(index)
                                {
                                    *seen = true;
                                }
                            }
                            if calls.is_empty() {
                                choice.delta.tool_calls = None;
                            }
                        }
                    }
                }
                yield serialize_chat_stream_response(response.clone());
            } else if let Some(error) = item.error {
                yield error;
            }
        }
        yield "[DONE]".to_string();
    }
}

fn serialize_chat_stream_response(response: CreateChatCompletionStreamResponse) -> String {
    let mut response = serde_json::to_value(response).expect("OpenAI response must serialize");
    if let Some(delta) = response
        .pointer_mut("/choices/0/delta")
        .and_then(serde_json::Value::as_object_mut)
    {
        delta
            .entry("reasoning_content")
            .or_insert(serde_json::Value::Null);
    }
    response.to_string()
}

#[allow(deprecated)]
pub(super) fn chat_logprobs(extras: Option<&ChunkExtras>) -> ChatChoiceLogprobs {
    let mut content = Vec::new();
    let Some(extras) = extras else {
        return ChatChoiceLogprobs {
            content: Some(content),
            refusal: None,
        };
    };
    let mut top_offset = 0usize;
    for (position, (&logprob, &token_id)) in
        extras.out_lp_val.iter().zip(&extras.out_lp_idx).enumerate()
    {
        let token = extras
            .out_lp_txt
            .get(position)
            .cloned()
            .unwrap_or_else(|| format!("token_id:{token_id}"));
        let top_len = extras.out_top_lens.get(position).copied().unwrap_or(0) as usize;
        let top_logprobs = extras.out_top_val[top_offset..]
            .iter()
            .zip(&extras.out_top_idx[top_offset..])
            .take(top_len)
            .enumerate()
            .map(|(offset, (&logprob, &id))| {
                let text = extras
                    .out_top_txt
                    .get(top_offset + offset)
                    .cloned()
                    .unwrap_or_else(|| format!("token_id:{id}"));
                TopLogprobs {
                    bytes: Some(text.as_bytes().to_vec()),
                    token: text,
                    logprob,
                }
            })
            .collect();
        top_offset = top_offset.saturating_add(top_len);
        content.push(ChatCompletionTokenLogprob {
            bytes: Some(token.as_bytes().to_vec()),
            token,
            logprob,
            token_id: u32::try_from(token_id).ok(),
            top_logprobs,
        });
    }
    ChatChoiceLogprobs {
        content: Some(content),
        refusal: None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::{chat_submitted, chunk, senders};
    use super::{chat_event_stream, chat_logprobs, unary_chat};
    use crate::api_server::guard::AbortGuard;
    use crate::message::response::ChunkExtras;
    use axum::http::StatusCode;
    use futures::StreamExt;

    #[test]
    fn chat_logprobs_use_dynamo_wire_types() {
        let extras = ChunkExtras {
            out_lp_val: vec![-0.25],
            out_lp_idx: vec![7],
            out_lp_txt: vec!["x".into()],
            out_top_val: vec![-0.25, -1.0],
            out_top_idx: vec![7, 8],
            out_top_lens: vec![2],
            out_top_txt: vec!["x".into(), "y".into()],
            ..Default::default()
        };
        let logprobs = chat_logprobs(Some(&extras));
        let token = &logprobs.content.unwrap()[0];
        assert_eq!(token.token, "x");
        assert_eq!(token.token_id, Some(7));
        assert_eq!(token.top_logprobs.len(), 2);
        assert_eq!(token.top_logprobs[1].token, "y");
    }

    #[tokio::test]
    async fn unary_chat_fans_in_choices_and_usage() {
        let (choice0, tx0) = chat_submitted(0, "r0");
        let (choice1, tx1) = chat_submitted(1, "r1");
        tx0.send(chunk("r0", "Paris", true)).await.unwrap();
        tx1.send(chunk("r1", "Paris", true)).await.unwrap();

        let response = unary_chat(
            vec![choice0, choice1],
            AbortGuard::new_empty(senders()),
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            None,
            None,
            None,
            true,
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert_eq!(value["choices"][1]["index"], 1);
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn unary_chat_separates_reasoning_content_with_parser_configured() {
        let (choice, tx) = chat_submitted(0, "r0");
        tx.send(chunk(
            "r0",
            "<think>because Paris is famous</think>Paris",
            true,
        ))
        .await
        .unwrap();

        let response = unary_chat(
            vec![choice],
            AbortGuard::new_empty(senders()),
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            None,
            Some("deepseek-r1".into()),
            None,
            true,
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "because Paris is famous"
        );
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert!(value["choices"][0]["message"]["reasoning_content"].is_string());
    }

    #[tokio::test]
    async fn streaming_chat_separates_reasoning_into_own_deltas() {
        let (choice, tx) = chat_submitted(0, "r0");
        // Force mode starts in reasoning, so the opener is stripped and the first
        // reasoning fragment streams immediately.
        tx.send(chunk("r0", "<think>be", false)).await.unwrap();
        tx.send(chunk("r0", "cause</think>Par", false))
            .await
            .unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            AbortGuard::new_empty(senders()),
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            true,
            None,
            Some("deepseek-r1".into()),
            None,
            None,
            false,
            true,
            None,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let first_reasoning: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let second_reasoning: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        let content: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[4]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            first_reasoning["choices"][0]["delta"]["reasoning_content"],
            "be"
        );
        assert!(first_reasoning["choices"][0]["delta"]["content"].is_null());
        assert_eq!(
            second_reasoning["choices"][0]["delta"]["reasoning_content"],
            "cause"
        );
        assert_eq!(content["choices"][0]["delta"]["content"], "Par");
        assert!(content["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(frames.len(), 7);
    }

    #[tokio::test]
    async fn streaming_chat_emits_role_deltas_usage_and_done() {
        let (choice, tx) = chat_submitted(0, "r0");
        tx.send(chunk("r0", "Par", false)).await.unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            AbortGuard::new_empty(senders()),
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            true,
            None,
            None,
            None,
            None,
            false,
            true,
            None,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        assert_eq!(frames.len(), 5);
        let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let delta: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        let usage: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert!(role["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(delta["choices"][0]["delta"]["content"], "Par");
        assert!(delta["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
        assert!(terminal["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(frames[4], "[DONE]");
    }
}
