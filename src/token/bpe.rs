use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::{
    gguf::header::{GgufMetadata, GgufMetadataValue},
    token::{
        pre_tokenizer::{PreTokenizer, create_pre_tokenizer},
        tokenizer::Tokenizer,
    },
};

pub struct BpeTokenizer {
    token_to_id: HashMap<Vec<u8>, u32>,
    id_to_token: HashMap<u32, Vec<u8>>,
    merges: HashMap<(Vec<u8>, Vec<u8>), usize>,
    eos_token_id: u32,
    bos_token_id: u32,
    pre_tokenizer: Box<dyn PreTokenizer>,
}

impl BpeTokenizer {
    pub fn new(metadata_kv: &GgufMetadata) -> Result<Self> {
        let tokens = match metadata_kv.get("tokenizer.ggml.tokens") {
            Some(GgufMetadataValue::Array(tokens)) => tokens,
            _ => bail!("Missing required key 'tokenizer.ggml.tokens' (expected Array)"),
        };

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (i, raw_token) in tokens.iter().enumerate() {
            if let GgufMetadataValue::String(token) = raw_token {
                let byte_token = token.as_bytes().to_vec();
                token_to_id.insert(byte_token.clone(), i as u32);
                id_to_token.insert(i as u32, byte_token);
            } else {
                bail!(
                    "Invalid value in 'tokenizer.ggml.tokens' at index {}: expected String",
                    i
                );
            }
        }

        let raw_merges = match metadata_kv.get("tokenizer.ggml.merges") {
            Some(GgufMetadataValue::Array(merges)) => merges,
            _ => bail!("Missing required key 'tokenizer.ggml.merges' (expected Array)"),
        };

        let mut merges = HashMap::new();

        for (i, raw_merge) in raw_merges.iter().enumerate() {
            if let GgufMetadataValue::String(merge) = raw_merge {
                let parts: Vec<&str> = merge.split(' ').collect();
                merges.insert(
                    (parts[0].as_bytes().to_vec(), parts[1].as_bytes().to_vec()),
                    i,
                );
            } else {
                bail!(
                    "Invalid value in 'tokenizer.ggml.merges' at index {}: expected String",
                    i
                );
            }
        }

        let eos_token_id = match metadata_kv.get("tokenizer.ggml.eos_token_id") {
            Some(GgufMetadataValue::U32(id)) => *id,
            _ => bail!("Missing required key 'tokenizer.ggml.eos_token_id' (expected U32)"),
        };
        let bos_token_id = match metadata_kv.get("tokenizer.ggml.bos_token_id") {
            Some(GgufMetadataValue::U32(id)) => *id,
            _ => bail!("Missing required key 'tokenizer.ggml.bos_token_id' (expected U32)"),
        };

        let pre_encode = match metadata_kv.get("tokenizer.ggml.pre") {
            Some(GgufMetadataValue::String(pre)) => pre,
            _ => bail!("Missing required key 'tokenizer.ggml.pre' (expected String)"),
        };

        let pre_tokenizer = create_pre_tokenizer(pre_encode)?;

        Ok(Self {
            token_to_id,
            id_to_token,
            merges,
            eos_token_id,
            bos_token_id,
            pre_tokenizer,
        })
    }
}

impl<'a> Tokenizer<'a, u8> for BpeTokenizer {
    fn pre_encode(&self, text: &'a str) -> Result<Vec<&'a [u8]>> {
        self.pre_tokenizer.pre_tokenize(text)
    }

    fn encode(&self, text: &'a str) -> Result<Vec<u32>> {
        let chunks = self.pre_encode(text)?;

        let mut ids = Vec::new();

        for chunk in chunks {
            let mut bytes: Vec<Vec<u8>> = chunk
                .iter()
                .map(|&c| {
                    let ch = byte_to_char(c);
                    let mut buf = [0; 4];
                    ch.encode_utf8(&mut buf).as_bytes().to_vec()
                })
                .collect();

            loop {
                let mut best_pair_index = None;
                let mut best_pair_priority = usize::MAX;

                for i in 0..bytes.len().saturating_sub(1) {
                    let pair = (bytes[i].clone(), bytes[i + 1].clone());

                    if let Some(&pair_priority) = self.merges.get(&pair) {
                        if pair_priority < best_pair_priority {
                            best_pair_index = Some(i);
                            best_pair_priority = pair_priority;
                        }
                    }
                }

                if let Some(best_pair_idx) = best_pair_index {
                    let right_bytes = bytes.remove(best_pair_idx + 1);
                    bytes[best_pair_idx].extend(right_bytes);
                } else {
                    break;
                }
            }

            for byte in bytes {
                if let Some(&id) = self.token_to_id.get(&byte) {
                    ids.push(id);
                } else {
                    eprintln!("Warning: Unknown token {:?}", byte);
                }
            }
        }

        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<u8> = ids
            .iter()
            .map(|id| self.id_to_token.get(id).unwrap().clone())
            .flatten()
            .collect();

        let s = String::from_utf8(tokens)?;
        let decoded_bytes: Vec<u8> = s
            .chars()
            .flat_map(|c| {
                if let Some(b) = char_to_byte(c) {
                    vec![b]
                } else {
                    let mut buf = [0; 4];
                    c.encode_utf8(&mut buf).as_bytes().to_vec()
                }
            })
            .collect();

        Ok(String::from_utf8_lossy(&decoded_bytes).into())
    }
}

fn byte_to_char(b: u8) -> char {
    let code = match b {
        33..=126 => b as u32,
        161..=172 => b as u32,
        174..=255 => b as u32,
        0..=32 => 256 + b as u32,
        127..=160 => 289 + (b - 127) as u32,
        173 => 323,
    };
    std::char::from_u32(code).unwrap()
}

fn char_to_byte(c: char) -> Option<u8> {
    let code = c as u32;
    Some(match code {
        33..=126 => code as u8,
        161..=172 => code as u8,
        174..=255 => code as u8,
        256..=288 => (code - 256) as u8,
        289..=322 => (127 + (code - 289)) as u8,
        323 => 173,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metadata() -> GgufMetadata {
        let mut metadata = GgufMetadata::new();

        // 0: <bos>
        // 1: <eos>
        // 2: h
        // 3: e
        // 4: l
        // 5: o
        // 6: he
        // 7: ll
        // 8: hell
        // 9: hello
        // 10: Ġ (space, which is \u{0120})
        // 11: Ġh
        // 12: あ
        let tokens = vec![
            GgufMetadataValue::String("<bos>".to_string()),
            GgufMetadataValue::String("<eos>".to_string()),
            GgufMetadataValue::String("h".to_string()),
            GgufMetadataValue::String("e".to_string()),
            GgufMetadataValue::String("l".to_string()),
            GgufMetadataValue::String("o".to_string()),
            GgufMetadataValue::String("he".to_string()),
            GgufMetadataValue::String("ll".to_string()),
            GgufMetadataValue::String("hell".to_string()),
            GgufMetadataValue::String("hello".to_string()),
            GgufMetadataValue::String("\u{0120}".to_string()),
            GgufMetadataValue::String("\u{0120}h".to_string()),
            GgufMetadataValue::String("あ".to_string()),
        ];
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::Array(tokens),
        );

        let merges = vec![
            GgufMetadataValue::String("h e".to_string()),
            GgufMetadataValue::String("l l".to_string()),
            GgufMetadataValue::String("he ll".to_string()),
            GgufMetadataValue::String("hell o".to_string()),
            GgufMetadataValue::String("\u{0120} h".to_string()),
        ];
        metadata.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufMetadataValue::Array(merges),
        );

        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufMetadataValue::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufMetadataValue::U32(1),
        );
        metadata.insert(
            "tokenizer.ggml.pre".to_string(),
            GgufMetadataValue::String("gpt-2".to_string()),
        );

        metadata
    }

    #[test]
    fn test_byte_char_roundtrip() {
        for b in 0..=255 {
            let c = byte_to_char(b);
            let b_back = char_to_byte(c);
            assert_eq!(b_back, Some(b), "Failed roundtrip for byte {}", b);
        }
    }

    #[test]
    fn test_bpe_tokenizer_creation_success() -> Result<()> {
        let metadata = create_test_metadata();
        let tokenizer = BpeTokenizer::new(&metadata)?;
        assert_eq!(tokenizer.bos_token_id, 0);
        assert_eq!(tokenizer.eos_token_id, 1);
        Ok(())
    }

    #[test]
    fn test_bpe_tokenizer_creation_missing_or_invalid_keys() {
        // Missing tokens
        {
            let mut metadata = create_test_metadata();
            metadata.remove("tokenizer.ggml.tokens");
            assert!(BpeTokenizer::new(&metadata).is_err());
        }

        // Invalid tokens type
        {
            let mut metadata = create_test_metadata();
            metadata.insert(
                "tokenizer.ggml.tokens".to_string(),
                GgufMetadataValue::U32(42),
            );
            assert!(BpeTokenizer::new(&metadata).is_err());
        }

        // Invalid token value type (e.g. non-string array elements)
        {
            let mut metadata = create_test_metadata();
            metadata.insert(
                "tokenizer.ggml.tokens".to_string(),
                GgufMetadataValue::Array(vec![GgufMetadataValue::U32(1)]),
            );
            assert!(BpeTokenizer::new(&metadata).is_err());
        }

        // Missing merges
        {
            let mut metadata = create_test_metadata();
            metadata.remove("tokenizer.ggml.merges");
            assert!(BpeTokenizer::new(&metadata).is_err());
        }

        // Invalid merges type
        {
            let mut metadata = create_test_metadata();
            metadata.insert(
                "tokenizer.ggml.merges".to_string(),
                GgufMetadataValue::U32(42),
            );
            assert!(BpeTokenizer::new(&metadata).is_err());
        }

        // Invalid merge value type
        {
            let mut metadata = create_test_metadata();
            metadata.insert(
                "tokenizer.ggml.merges".to_string(),
                GgufMetadataValue::Array(vec![GgufMetadataValue::U32(1)]),
            );
            assert!(BpeTokenizer::new(&metadata).is_err());
        }

        // Missing bos_token_id
        {
            let mut metadata = create_test_metadata();
            metadata.remove("tokenizer.ggml.bos_token_id");
            assert!(BpeTokenizer::new(&metadata).is_err());
        }

        // Missing eos_token_id
        {
            let mut metadata = create_test_metadata();
            metadata.remove("tokenizer.ggml.eos_token_id");
            assert!(BpeTokenizer::new(&metadata).is_err());
        }

        // Missing pre
        {
            let mut metadata = create_test_metadata();
            metadata.remove("tokenizer.ggml.pre");
            assert!(BpeTokenizer::new(&metadata).is_err());
        }

        // Unsupported pre-tokenizer
        {
            let mut metadata = create_test_metadata();
            metadata.insert(
                "tokenizer.ggml.pre".to_string(),
                GgufMetadataValue::String("invalid-pretokenizer".to_string()),
            );
            assert!(BpeTokenizer::new(&metadata).is_err());
        }
    }

    #[test]
    fn test_bpe_tokenizer_encode_decode_basic() -> Result<()> {
        let metadata = create_test_metadata();
        let tokenizer = BpeTokenizer::new(&metadata)?;

        // Test basic encoding
        let encoded = tokenizer.encode("hello")?;
        assert_eq!(encoded, vec![9]); // "hello" -> ID 9

        // Test decode
        let decoded = tokenizer.decode(&encoded)?;
        assert_eq!(decoded, "hello");

        // Test partial encode
        let encoded_partial = tokenizer.encode("he")?;
        assert_eq!(encoded_partial, vec![6]); // "he" -> ID 6
        assert_eq!(tokenizer.decode(&encoded_partial)?, "he");

        Ok(())
    }

    #[test]
    fn test_bpe_tokenizer_encode_decode_with_spaces() -> Result<()> {
        let metadata = create_test_metadata();
        let tokenizer = BpeTokenizer::new(&metadata)?;

        // Test encode with space
        // " hello" has a leading space.
        let encoded = tokenizer.encode(" hello")?;
        assert_eq!(encoded, vec![10, 9]);

        let decoded = tokenizer.decode(&encoded)?;
        assert_eq!(decoded, " hello");

        // Test encode with space followed by h, but no e (so no "h e" merge)
        let encoded_h = tokenizer.encode(" h")?;
        assert_eq!(encoded_h, vec![11]); // " h" -> ID 11
        assert_eq!(tokenizer.decode(&encoded_h)?, " h");

        Ok(())
    }

    #[test]
    fn test_bpe_tokenizer_decode_unmapped_char() -> Result<()> {
        let metadata = create_test_metadata();
        let tokenizer = BpeTokenizer::new(&metadata)?;

        // "あ" is ID 12.
        // It cannot be directly encoded because BPE encodes inputs via byte_to_char mappings.
        // However, decoding ID 12 should correctly fallback to its UTF-8 representation
        // since char_to_byte('あ') returns None.
        let decoded = tokenizer.decode(&[12])?;
        assert_eq!(decoded, "あ");

        Ok(())
    }

    #[test]
    fn test_bpe_tokenizer_unknown_token() -> Result<()> {
        let metadata = create_test_metadata();
        let tokenizer = BpeTokenizer::new(&metadata)?;

        // "x" is not in the vocabulary.
        // It should warning log but skip it.
        let encoded = tokenizer.encode("x")?;
        assert!(encoded.is_empty());

        Ok(())
    }
}
