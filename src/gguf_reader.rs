use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Result, bail};
use memmap2::MmapOptions;

use crate::binary_reader::BReader;

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_CURRENT_VERSION: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Count = 40,
}

impl TryFrom<u32> for GgmlType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            9 => Ok(GgmlType::Q8_1),
            10 => Ok(GgmlType::Q2K),
            11 => Ok(GgmlType::Q3K),
            12 => Ok(GgmlType::Q4K),
            13 => Ok(GgmlType::Q5K),
            14 => Ok(GgmlType::Q6K),
            15 => Ok(GgmlType::Q8K),
            16 => Ok(GgmlType::Iq2Xxs),
            17 => Ok(GgmlType::Iq2Xs),
            18 => Ok(GgmlType::Iq3Xxs),
            19 => Ok(GgmlType::Iq1S),
            20 => Ok(GgmlType::Iq4Nl),
            21 => Ok(GgmlType::Iq3S),
            22 => Ok(GgmlType::Iq2S),
            23 => Ok(GgmlType::Iq4Xs),
            24 => Ok(GgmlType::I8),
            25 => Ok(GgmlType::I16),
            26 => Ok(GgmlType::I32),
            27 => Ok(GgmlType::I64),
            28 => Ok(GgmlType::F64),
            29 => Ok(GgmlType::Iq1M),
            30 => Ok(GgmlType::Bf16),
            34 => Ok(GgmlType::Tq1_0),
            35 => Ok(GgmlType::Tq2_0),
            39 => Ok(GgmlType::Mxfp4),
            40 => Ok(GgmlType::Count),
            _ => bail!("Unsupported ggml type: {}", value),
        }
    }
}

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
    pub metadata_kv: HashMap<String, GgufMetadataValue>,
    pub architecture: String,
    pub quantization_version: Option<u32>,
    pub alignment: u32,
}

pub struct GgufTensorInfo {
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

pub struct GgufFile {
    pub header: GgufHeader,
    pub tensor_infos: HashMap<String, GgufTensorInfo>,
    pub _padding: Vec<u8>,
    pub tensor_data: Vec<u8>,
}

pub struct GgufReader {
    pub b_reader: BReader,
}

impl<'a> GgufReader {
    pub fn new(file_path: &'a Path) -> Result<Self> {
        let file = File::open(file_path)?;
        let buffer = unsafe { MmapOptions::new().map(&file)? };
        let b_reader = BReader::new(buffer);

        Ok(Self { b_reader })
    }

    pub fn read(&mut self) -> Result<GgufFile> {
        let header = self.read_header()?;
        let tensor_infos = self.read_tensor_infos(header.tensor_count)?;

        let is_quantized = tensor_infos.values().any(|info| {
            !matches!(
                info.ggml_type,
                GgmlType::F32
                    | GgmlType::F16
                    | GgmlType::F64
                    | GgmlType::I8
                    | GgmlType::I16
                    | GgmlType::I32
                    | GgmlType::I64
                    | GgmlType::Bf16
            )
        });

        if is_quantized && header.quantization_version.is_none() {
            bail!(
                "Missing required key 'general.quantization_version' (required because the model contains quantized tensors)"
            );
        }

        let _padding = self.read_padding(header.alignment as usize)?;
        let tensor_data = self.read_tensor_data()?;

        Ok(GgufFile {
            header,
            tensor_infos,
            _padding,
            tensor_data,
        })
    }

    fn read_header(&mut self) -> Result<GgufHeader> {
        let magic = self.b_reader.read_u32()?;

        if magic != GGUF_MAGIC {
            bail!(
                "Invalid GGUF magic number: expected 0x{:08X}, found 0x{:08X}",
                GGUF_MAGIC,
                magic
            );
        }

        let version = self.b_reader.read_u32()?;

        if version != GGUF_CURRENT_VERSION {
            bail!(
                "Unsupported GGUF version: found {}, expected {}",
                version,
                GGUF_CURRENT_VERSION
            );
        }

        let tensor_count = self.b_reader.read_u64()?;
        let metadata_kv_count = self.b_reader.read_u64()?;
        let mut metadata_kv = HashMap::new();

        for _ in 0..metadata_kv_count {
            let key = self.b_reader.read_string()?;
            let value_type = self.b_reader.read_u32()?;
            let value = self.read_metadata_value(value_type)?;

            metadata_kv.insert(key, value);
        }

        let architecture = match metadata_kv.get("general.architecture") {
            Some(GgufMetadataValue::String(arch)) => {
                if !arch
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                {
                    bail!(
                        "Invalid 'general.architecture' value '{}': must contain only lowercase ASCII alphanumeric characters",
                        arch
                    );
                }
                arch.clone()
            }
            _ => bail!("Missing required key 'general.architecture' (expected string)"),
        };

        let quantization_version = match metadata_kv.get("general.quantization_version") {
            Some(GgufMetadataValue::U32(version)) => Some(*version),
            None => None,
            _ => bail!("Invalid key 'general.quantization_version': expected U32"),
        };

        let alignment = match metadata_kv.get("general.alignment") {
            Some(GgufMetadataValue::U32(alignment)) => {
                if alignment % 8 != 0 {
                    bail!(
                        "Invalid 'general.alignment' value: {} (must be a multiple of 8)",
                        alignment
                    );
                }
                *alignment
            }
            Some(_) => bail!("Invalid key 'general.alignment': expected U32"),
            _ => 32,
        };

        Ok(GgufHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
            metadata_kv,
            architecture,
            quantization_version,
            alignment,
        })
    }

    fn read_metadata_value(&mut self, value_type: u32) -> Result<GgufMetadataValue> {
        Ok(match value_type {
            0 => GgufMetadataValue::U8(self.b_reader.read_u8()?),
            1 => GgufMetadataValue::I8(self.b_reader.read_i8()?),
            2 => GgufMetadataValue::U16(self.b_reader.read_u16()?),
            3 => GgufMetadataValue::I16(self.b_reader.read_i16()?),
            4 => GgufMetadataValue::U32(self.b_reader.read_u32()?),
            5 => GgufMetadataValue::I32(self.b_reader.read_i32()?),
            6 => GgufMetadataValue::F32(self.b_reader.read_f32()?),
            7 => GgufMetadataValue::Boolean(self.b_reader.read_boolean()?),
            8 => GgufMetadataValue::String(self.b_reader.read_string()?),
            9 => {
                let array_type = self.b_reader.read_u32()?;
                let array_length = self.b_reader.read_u64()?;
                let mut values: Vec<GgufMetadataValue> = Vec::new();

                for _ in 0..array_length {
                    values.push(self.read_metadata_value(array_type)?);
                }

                GgufMetadataValue::Array(values)
            }
            10 => GgufMetadataValue::U64(self.b_reader.read_u64()?),
            11 => GgufMetadataValue::I64(self.b_reader.read_i64()?),
            12 => GgufMetadataValue::F64(self.b_reader.read_f64()?),
            _ => bail!("Unknown GGUF metadata value type ID: {}", value_type),
        })
    }

    fn read_tensor_infos(&mut self, tensor_count: u64) -> Result<HashMap<String, GgufTensorInfo>> {
        let mut tensor_infos = HashMap::new();

        for _ in 0..tensor_count {
            let name = self.b_reader.read_string()?;
            let n_dimensions = self.b_reader.read_u32()?;
            let mut dimensions = Vec::new();

            for _ in 0..n_dimensions {
                dimensions.push(self.b_reader.read_u64()?);
            }

            let ggml_type: GgmlType = self.b_reader.read_u32()?.try_into()?;
            let offset = self.b_reader.read_u64()?;

            tensor_infos.insert(
                name,
                GgufTensorInfo {
                    n_dimensions,
                    dimensions,
                    ggml_type,
                    offset,
                },
            );
        }

        Ok(tensor_infos)
    }

    fn read_padding(&mut self, alignment: usize) -> Result<Vec<u8>> {
        let offset = self.b_reader.position;
        let align_offset = offset + (alignment - (offset % alignment)) % alignment;

        Ok(self.b_reader.read_bytes(align_offset - offset)?.to_vec())
    }

    fn read_tensor_data(&mut self) -> Result<Vec<u8>> {
        let remaining_bytes = self.b_reader.buffer.len() - self.b_reader.position;

        Ok(self.b_reader.read_bytes(remaining_bytes)?.to_vec())
    }
}
