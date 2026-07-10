use anyhow::{Result, bail};
use fancy_regex::Regex;

pub trait PreTokenizer: Send + Sync {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a [u8]>>;
}

pub struct RegexPreTokenizer {
    regex: Regex,
}

impl RegexPreTokenizer {
    pub fn new(pattern: &str) -> Result<Self> {
        let regex = Regex::new(pattern)?;
        Ok(Self { regex })
    }
}

impl PreTokenizer for RegexPreTokenizer {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a [u8]>> {
        let mut chunks = Vec::new();
        for m in self.regex.find_iter(text) {
            let m = m?;
            chunks.push(m.as_str().as_bytes());
        }
        Ok(chunks)
    }
}

pub fn create_pre_tokenizer(name: &str) -> Result<Box<dyn PreTokenizer>> {
    match name {
        "llama-bpe" => {
            let pattern = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
            Ok(Box::new(RegexPreTokenizer::new(pattern)?))
        }
        "gpt2" | "gpt-2" => {
            let pattern =
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
            Ok(Box::new(RegexPreTokenizer::new(pattern)?))
        }
        _ => bail!("Unsupported pre-tokenizer '{}'", name),
    }
}
