use std::path::Path;

use anyhow::Result;

use crate::{
    gguf::reader::GgufReader,
    token::{bpe::BpeTokenizer, tokenizer::Tokenizer},
};

mod binary_reader;
mod gguf;
mod token;

fn main() -> Result<()> {
    let path = Path::new("data/Llama-3.2-3B-Instruct-Q8_0.gguf");
    let mut reader = GgufReader::new(&path)?;

    let gguf = reader.read()?;
    let tokenizer = BpeTokenizer::new(&gguf.header.metadata_kv)?;

    let encoded = tokenizer.encode("Hello, World!")?;

    println!("{:?}", encoded);
    println!("{}", tokenizer.decode(&encoded)?);

    Ok(())
}
