use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Result, bail};
use memmap2::MmapOptions;

use crate::{
    binary_reader::BReader,
    gguf::{
        ggml::GgmlType,
        header::{GGUF_CURRENT_VERSION, GGUF_MAGIC, GgufHeader, GgufMetadataValue},
    },
};

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_read_minimal_gguf() -> Result<()> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x46554747u32.to_le_bytes()); // magic
        bytes.extend_from_slice(&3u32.to_le_bytes());          // version
        bytes.extend_from_slice(&0u64.to_le_bytes());          // tensor count
        bytes.extend_from_slice(&1u64.to_le_bytes());          // metadata KV count

        // Metadata KV: "general.architecture" = "llama"
        let key = "general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());

        bytes.extend_from_slice(&8u32.to_le_bytes()); // Type ID 8: String

        let val = "llama";
        bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
        bytes.extend_from_slice(val.as_bytes());

        // Pad the file bytes to align with 32-byte boundary, and add dummy tensor data
        let alignment = 32;
        let padding_needed = (alignment - (bytes.len() % alignment)) % alignment;
        bytes.extend(std::iter::repeat(0).take(padding_needed));
        bytes.extend_from_slice(b"tensor-payload");

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("minimal_test.gguf");
        fs::write(&file_path, bytes)?;

        let mut reader = GgufReader::new(&file_path)?;
        let gguf_file = reader.read()?;

        assert_eq!(gguf_file.header.magic, 0x46554747);
        assert_eq!(gguf_file.header.version, 3);
        assert_eq!(gguf_file.header.architecture, "llama");
        assert_eq!(gguf_file.header.alignment, 32);
        assert_eq!(gguf_file.tensor_infos.len(), 0);
        assert_eq!(gguf_file.tensor_data, b"tensor-payload");

        // Clean up
        let _ = fs::remove_file(file_path);
        Ok(())
    }
}

