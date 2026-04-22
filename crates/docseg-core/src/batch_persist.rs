//! Batch ↔ zip serialization plus schema migration scaffold.
//!
//! Layout:
//!   manifest.json              — schema_version + format marker
//!   batch.json                 — Batch without image bytes
//!   images/page_NNNN.bin       — sidecar raw image bytes

use std::io::{Cursor, Read, Write};

use sha2::{Digest, Sha256};
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::batch::{Batch, CURRENT_SCHEMA_VERSION};
use crate::CoreError;

/// Serialize a `Batch` + every page's image bytes into a zip archive.
pub fn to_zip(batch: &Batch) -> Result<Vec<u8>, CoreError> {
    let mut buf = Cursor::new(Vec::new());
    {
        let mut zw = ZipWriter::new(&mut buf);
        let opts: FileOptions = FileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(0o644);

        // Write manifest.json first (small, top of archive).
        let manifest_body = format!(
            "{{\"schema_version\":{},\"format\":\"docseg-batch-zip\"}}",
            batch.schema_version
        );
        zw.start_file("manifest.json", opts)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("zip manifest: {e}"),
            })?;
        zw.write_all(manifest_body.as_bytes()).map_err(io_err)?;

        // Write batch.json (pages included, image_bytes skipped via #[serde(skip)]).
        let batch_json = serde_json::to_vec_pretty(batch).map_err(|e| CoreError::Postprocess {
            reason: format!("batch json: {e}"),
        })?;
        zw.start_file("batch.json", opts)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("zip batch.json: {e}"),
            })?;
        zw.write_all(&batch_json).map_err(io_err)?;

        // Sidecar images.
        for (i, page) in batch.pages.iter().enumerate() {
            let name = format!("images/page_{:04}.bin", i + 1);
            zw.start_file(&name, opts)
                .map_err(|e| CoreError::Postprocess {
                    reason: format!("zip {name}: {e}"),
                })?;
            zw.write_all(&page.image_bytes).map_err(io_err)?;
        }
        zw.finish().map_err(|e| CoreError::Postprocess {
            reason: format!("zip finish: {e}"),
        })?;
    }
    Ok(buf.into_inner())
}

/// Deserialize a zip produced by `to_zip`.
pub fn from_zip(bytes: &[u8]) -> Result<Batch, CoreError> {
    let mut zr = ZipArchive::new(Cursor::new(bytes)).map_err(|e| CoreError::Postprocess {
        reason: format!("open zip: {e}"),
    })?;

    let mut manifest = String::new();
    {
        let mut f = zr
            .by_name("manifest.json")
            .map_err(|e| CoreError::Postprocess {
                reason: format!("manifest.json: {e}"),
            })?;
        f.read_to_string(&mut manifest).map_err(io_err)?;
    }
    let schema_version = extract_schema_version(&manifest)?;
    if schema_version > CURRENT_SCHEMA_VERSION {
        return Err(CoreError::Postprocess {
            reason: format!(
                "schema version {schema_version} too new; this build supports {CURRENT_SCHEMA_VERSION}. Run a newer docseg to migrate."
            ),
        });
    }

    let mut batch_json = Vec::new();
    {
        let mut f = zr
            .by_name("batch.json")
            .map_err(|e| CoreError::Postprocess {
                reason: format!("batch.json: {e}"),
            })?;
        f.read_to_end(&mut batch_json).map_err(io_err)?;
    }
    let mut batch: Batch =
        serde_json::from_slice(&batch_json).map_err(|e| CoreError::Postprocess {
            reason: format!("parse batch.json: {e}"),
        })?;

    // Rehydrate sidecar images and verify SHA-256 drift.
    for (i, page) in batch.pages.iter_mut().enumerate() {
        let name = format!("images/page_{:04}.bin", i + 1);
        let mut image_bytes = Vec::new();
        {
            let mut f = zr.by_name(&name).map_err(|e| CoreError::Postprocess {
                reason: format!("{name}: {e}"),
            })?;
            f.read_to_end(&mut image_bytes).map_err(io_err)?;
        }
        let mut sha = [0u8; 32];
        sha.copy_from_slice(&Sha256::digest(&image_bytes));
        if sha != page.image_sha256 {
            // Don't fail the whole load; flag the page instead.
            page.status = crate::batch::PageStatus::Flagged;
        }
        page.image_bytes = image_bytes;
    }

    batch = migrate(batch, schema_version)?;
    Ok(batch)
}

/// Apply the migration chain from `from` up to `CURRENT_SCHEMA_VERSION`.
/// v1 has no prior schema, so this is a pass-through for the current version.
pub fn migrate(batch: Batch, from: u32) -> Result<Batch, CoreError> {
    if from == CURRENT_SCHEMA_VERSION {
        return Ok(batch);
    }
    Err(CoreError::Postprocess {
        reason: format!("no migration path from schema {from} to {CURRENT_SCHEMA_VERSION}"),
    })
}

fn extract_schema_version(manifest: &str) -> Result<u32, CoreError> {
    // Manifest is tiny, controlled JSON; regex-free parse is fine.
    let key = "\"schema_version\":";
    let start = manifest.find(key).ok_or_else(|| CoreError::Postprocess {
        reason: "manifest missing schema_version".into(),
    })? + key.len();
    let tail = manifest[start..]
        .trim_start()
        .split(&[',', '}'][..])
        .next()
        .unwrap_or("")
        .trim();
    tail.parse::<u32>().map_err(|e| CoreError::Postprocess {
        reason: format!("parse schema_version: {e}"),
    })
}

fn io_err(e: std::io::Error) -> CoreError {
    CoreError::Postprocess {
        reason: format!("zip io: {e}"),
    }
}

#[cfg(test)]
mod tests;
