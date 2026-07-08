use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
}
