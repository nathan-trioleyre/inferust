use std::collections::HashMap;

pub const GGUF_MAGIC: u32 = 0x46554747;
pub const GGUF_CURRENT_VERSION: u32 = 3;

pub type GgufMetadata = HashMap<String, GgufMetadataValue>;

#[derive(Debug)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Boolean(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug)]
pub struct GgufHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata_kv: GgufMetadata,
    pub architecture: String,
    pub quantization_version: Option<u32>,
    pub alignment: u32,
}
