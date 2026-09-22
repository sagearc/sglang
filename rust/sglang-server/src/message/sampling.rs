//! Native sampling input parsing and preferred-default precedence.

use std::collections::BTreeSet;
use std::fmt;

use serde::de::value::{MapAccessDeserializer, SeqAccessDeserializer};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

pub use sglang_renderer::SamplingParams;

/// Native request parameters and the keys supplied by the caller. Presence
/// matters when preferred defaults are applied, even for explicit nulls.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParamsEntry {
    pub params: SamplingParams,
    explicit_fields: BTreeSet<String>,
}

/// The `/generate` body's `sampling_params`: one object (broadcast to every
/// prompt) or a list of them (one per prompt), fanned out by `GenerateBody::into_requests`.
///
/// Hand-written `Deserialize` rather than `#[serde(untagged)]`: untagged buffers
/// the input and, on failure, reports only "data did not match any variant" —
/// losing the field-level message ("unknown field `temperature`, expected one of
/// …") that makes a typo actionable. Object-vs-list is unambiguous here, so a
/// single `deserialize_any` dispatch keeps the inner error verbatim.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingParamsInput {
    /// Boxed: `SamplingParams` is ~440 bytes, so an inline variant would make
    /// every `GenerateBody` that big regardless of which form arrived.
    One(Box<SamplingParamsEntry>),
    Many(Vec<SamplingParamsEntry>),
}

impl<'de> Deserialize<'de> for SamplingParamsInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InputVisitor;

        impl<'de> Visitor<'de> for InputVisitor {
            type Value = SamplingParamsInput;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a sampling_params object, or a list of them (one per prompt)")
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                let value = serde_json::Value::deserialize(MapAccessDeserializer::new(map))?;
                sampling_params_from_value(value)
                    .map(|p| SamplingParamsInput::One(Box::new(p)))
                    .map_err(serde::de::Error::custom)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                let values =
                    Vec::<serde_json::Value>::deserialize(SeqAccessDeserializer::new(seq))?;
                values
                    .into_iter()
                    .map(sampling_params_from_value)
                    .collect::<Result<Vec<_>, _>>()
                    .map(SamplingParamsInput::Many)
                    .map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_any(InputVisitor)
    }
}

fn sampling_params_from_value(value: serde_json::Value) -> Result<SamplingParamsEntry, String> {
    let explicit_fields = value
        .as_object()
        .ok_or_else(|| "sampling_params must be an object".to_string())?
        .keys()
        .cloned()
        .collect();
    let params = serde_json::from_value(value).map_err(|e| e.to_string())?;
    Ok(SamplingParamsEntry {
        params,
        explicit_fields,
    })
}

impl SamplingParamsInput {
    /// Merge launch-time preferred params beneath request params. A request key
    /// wins even when it explicitly carries the type's default or null.
    pub fn apply_preferred(&mut self, preferred: &serde_json::Value) -> Result<(), String> {
        match self {
            Self::One(params) => apply_preferred_to_one(params, preferred),
            Self::Many(params) => params
                .iter_mut()
                .try_for_each(|params| apply_preferred_to_one(params, preferred)),
        }
    }

    pub fn from_preferred(preferred: &serde_json::Value) -> Result<Self, String> {
        sampling_params_from_value(preferred.clone()).map(|params| Self::One(Box::new(params)))
    }
}

fn apply_preferred_to_one(
    params: &mut SamplingParamsEntry,
    preferred: &serde_json::Value,
) -> Result<(), String> {
    let mut merged = preferred
        .as_object()
        .ok_or_else(|| "preferred_sampling_params must be a JSON object".to_string())?
        .clone();
    let request_value = serde_json::to_value(&params.params).map_err(|e| e.to_string())?;
    let request = request_value
        .as_object()
        .ok_or_else(|| "SamplingParams did not serialize as an object".to_string())?;
    for field in &params.explicit_fields {
        if let Some(value) = request.get(field) {
            merged.insert(field.clone(), value.clone());
        }
    }
    *params = sampling_params_from_value(serde_json::Value::Object(merged))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preferred_params_fill_only_omitted_request_fields() {
        let preferred = serde_json::json!({
            "temperature": 0.25,
            "top_p": 0.75,
            "max_new_tokens": 4096
        });
        let mut input: SamplingParamsInput =
            serde_json::from_str(r#"{"temperature": 1.0, "top_p": null}"#).unwrap();
        input.apply_preferred(&preferred).unwrap();
        let SamplingParamsInput::One(params) = input else {
            panic!("expected scalar params")
        };
        assert_eq!(params.params.temperature, 1.0, "explicit default wins");
        assert_eq!(
            params.params.top_p, 1.0,
            "explicit null keeps the type default"
        );
        assert_eq!(
            params.params.max_new_tokens,
            Some(4096),
            "omitted uses preferred"
        );
    }

    #[test]
    fn preferred_params_apply_to_every_batched_object() {
        let preferred = serde_json::json!({"temperature": 0.25, "top_p": 0.75});
        let mut input: SamplingParamsInput =
            serde_json::from_str(r#"[{"temperature": 0.5}, {"top_p": 0.9}]"#).unwrap();
        input.apply_preferred(&preferred).unwrap();
        let SamplingParamsInput::Many(params) = input else {
            panic!("expected batched params")
        };
        assert_eq!(
            (params[0].params.temperature, params[0].params.top_p),
            (0.5, 0.75)
        );
        assert_eq!(
            (params[1].params.temperature, params[1].params.top_p),
            (0.25, 0.9)
        );
    }
}
