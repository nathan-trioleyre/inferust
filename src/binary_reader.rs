use anyhow::{Result, bail};
use memmap2::Mmap;

pub struct BReader {
    pub buffer: Mmap,
    pub position: usize,
}

impl BReader {
    pub fn new(buffer: Mmap) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        if let Some(byte) = self.buffer.get(self.position..self.position + 1) {
            self.position += 1;
            return Ok(u8::from_le_bytes(byte.try_into()?));
        }

        bail!("Failed to read 1 byte at position {}", self.position);
    }

    pub fn read_i8(&mut self) -> Result<i8> {
        if let Some(byte) = self.buffer.get(self.position..self.position + 1) {
            self.position += 1;
            return Ok(i8::from_le_bytes(byte.try_into()?));
        }

        bail!("Failed to read 1 byte at position {}", self.position);
    }

    pub fn read_u16(&mut self) -> Result<u16> {
        if let Some(byte) = self.buffer.get(self.position..self.position + 2) {
            self.position += 2;
            return Ok(u16::from_le_bytes(byte.try_into()?));
        }

        bail!("Failed to read 2 bytes at position {}", self.position);
    }

    pub fn read_i16(&mut self) -> Result<i16> {
        if let Some(byte) = self.buffer.get(self.position..self.position + 2) {
            self.position += 2;
            return Ok(i16::from_le_bytes(byte.try_into()?));
        }

        bail!("Failed to read 2 bytes at position {}", self.position);
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        if let Some(bytes) = self.buffer.get(self.position..self.position + 4) {
            self.position += 4;
            return Ok(u32::from_le_bytes(bytes.try_into()?));
        }

        bail!("Failed to read 4 bytes at position {}", self.position);
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        if let Some(bytes) = self.buffer.get(self.position..self.position + 4) {
            self.position += 4;
            return Ok(i32::from_le_bytes(bytes.try_into()?));
        }

        bail!("Failed to read 4 bytes at position {}", self.position);
    }

    pub fn read_f32(&mut self) -> Result<f32> {
        if let Some(bytes) = self.buffer.get(self.position..self.position + 4) {
            self.position += 4;
            return Ok(f32::from_le_bytes(bytes.try_into()?));
        }

        bail!("Failed to read 4 bytes at position {}", self.position);
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        if let Some(bytes) = self.buffer.get(self.position..self.position + 8) {
            self.position += 8;
            return Ok(u64::from_le_bytes(bytes.try_into()?));
        }

        bail!("Failed to read 8 bytes at position {}", self.position);
    }

    pub fn read_i64(&mut self) -> Result<i64> {
        if let Some(bytes) = self.buffer.get(self.position..self.position + 8) {
            self.position += 8;
            return Ok(i64::from_le_bytes(bytes.try_into()?));
        }

        bail!("Failed to read 8 bytes at position {}", self.position);
    }

    pub fn read_f64(&mut self) -> Result<f64> {
        if let Some(bytes) = self.buffer.get(self.position..self.position + 8) {
            self.position += 8;
            return Ok(f64::from_le_bytes(bytes.try_into()?));
        }

        bail!("Failed to read 8 bytes at position {}", self.position);
    }

    pub fn read_boolean(&mut self) -> Result<bool> {
        Ok(self.read_u8()? == 1)
    }

    pub fn read_string(&mut self) -> Result<String> {
        let length = self.read_u64()? as usize;

        if let Some(bytes) = self.buffer.get(self.position..self.position + length) {
            self.position += length;

            let s = std::str::from_utf8(bytes)?;
            return Ok(s.to_string());
        }

        bail!(
            "Failed to read string of length {} at position {}",
            length,
            self.position
        );
    }

    pub fn read_bytes(&mut self, bytes_count: usize) -> Result<&[u8]> {
        if let Some(bytes) = self.buffer.get(self.position..self.position + bytes_count) {
            self.position += bytes_count;
            return Ok(bytes);
        }

        bail!(
            "Failed to read {} bytes at position {}",
            bytes_count,
            self.position
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::MmapMut;

    #[test]
    fn test_read_numeric_and_string() -> Result<()> {
        let mut mmap = MmapMut::map_anon(100)?;
        mmap[0] = 42;
        mmap[1] = -10i8 as u8;
        mmap[2..4].copy_from_slice(&1000u16.to_le_bytes());
        mmap[4..8].copy_from_slice(&0x12345678u32.to_le_bytes());
        mmap[8..12].copy_from_slice(&1.5f32.to_le_bytes());
        mmap[12..20].copy_from_slice(&5u64.to_le_bytes());
        mmap[20..25].copy_from_slice(b"hello");

        let readonly_mmap = mmap.make_read_only()?;
        let mut reader = BReader::new(readonly_mmap);

        assert_eq!(reader.read_u8()?, 42);
        assert_eq!(reader.read_i8()?, -10);
        assert_eq!(reader.read_u16()?, 1000);
        assert_eq!(reader.read_u32()?, 0x12345678);
        assert_eq!(reader.read_f32()?, 1.5);
        assert_eq!(reader.read_string()?, "hello");
        assert_eq!(reader.position, 25);

        // Test out of bounds
        assert!(reader.read_bytes(100).is_err());

        Ok(())
    }
}
