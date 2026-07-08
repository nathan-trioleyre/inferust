use std::path::Path;

use anyhow::Result;

use crate::gguf_reader::GgufReader;

mod binary_reader;
mod gguf_reader;

fn main() -> Result<()> {
    let path = Path::new("data/qwen1_5-0_5b-chat-q4_k_m.gguf");
    let mut reader = GgufReader::new(&path)?;

	let gguf = reader.read()?;

	println!("{:?}", gguf.header.metadata_kv.keys());

    Ok(())
}
