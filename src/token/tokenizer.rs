use anyhow::Result;

pub trait Tokenizer<'a, T> {
    fn pre_encode(&self, text: &'a str) -> Result<Vec<&'a [T]>>;
    fn encode(&self, text: &'a str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
}
