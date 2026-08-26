#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const SOURCE_REPOSITORY = "https://github.com/aster-droide/feline-age-prediction";
const SOURCE_COMMIT = "3d02295bef1500d2b2500a124596f77010181391";
const SOURCE_PREFIX = "dataset/raw_audio/AudioCropped/";
const EXPECTED_EMBEDDING_BLOB = "947a7c9baa983c18009f2a85bc12d95e51fbd48b";
const CONCURRENCY = 12;

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), "..");
const rawRoot = path.join(
  repoRoot,
  "data",
  "meowagenet",
  `official-${SOURCE_COMMIT.slice(0, 12)}`,
  "AudioCropped",
);
const metadataRoot = path.join(repoRoot, "metadata", "datasets", "meowagenet");

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--embedding-csv") {
      result.embeddingCsv = path.resolve(argv[index + 1]);
      index += 1;
    } else if (argv[index] === "--verify-only") {
      result.verifyOnly = true;
    } else {
      throw new Error(`Unknown argument: ${argv[index]}`);
    }
  }
  return result;
}

async function fetchWithRetry(url, responseType = "buffer", attempts = 4) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: {
          Accept: "application/vnd.github+json",
          "User-Agent": "animal-fyp-meowagenet-manifest",
        },
        signal: AbortSignal.timeout(30_000),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      if (responseType === "json") {
        return await response.json();
      }
      return Buffer.from(await response.arrayBuffer());
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await new Promise((resolve) => setTimeout(resolve, 500 * 2 ** (attempt - 1)));
      }
    }
  }
  throw new Error(`Failed to fetch ${url}: ${lastError?.message ?? lastError}`);
}

function hash(algorithm, buffer) {
  return createHash(algorithm).update(buffer).digest("hex");
}

function gitBlobHash(buffer) {
  const header = Buffer.from(`blob ${buffer.length}\0`, "utf8");
  return createHash("sha1").update(header).update(buffer).digest("hex");
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const stringValue = String(value);
  if (/[",\r\n]/.test(stringValue)) {
    return `"${stringValue.replaceAll('"', '""')}"`;
  }
  return stringValue;
}

function toCsv(rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function parseSimpleCsv(text) {
  const lines = text.trimEnd().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    if (values.length !== headers.length) {
      throw new Error("Embedding CSV contains an unsupported quoted or multiline row");
    }
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function parseFilename(filename) {
  const match = filename.match(/^(.+?)-(\d{3}[A-Z]{1,2})(?=[-_.])/);
  if (!match) {
    throw new Error(`Cannot parse age/cat ID from filename: ${filename}`);
  }

  const ageLabel = match[1];
  const sourceCatToken = match[2];
  const publishedCatId = sourceCatToken === "020AB" ? "020A" : sourceCatToken;

  const ageMatch = ageLabel.match(/^(\d+(?:\.\d+)?)Y(?:-(\d+)(month|wks|week))?$/);
  if (!ageMatch) {
    throw new Error(`Cannot parse age label from filename: ${filename}`);
  }
  let ageYears = Number(ageMatch[1]);
  if (ageMatch[2] && ageMatch[3] === "month") {
    ageYears += Number(ageMatch[2]) / 12;
  } else if (ageMatch[2]) {
    ageYears += Number(ageMatch[2]) / 52;
  }
  const ageGroup = ageYears < 0.5 ? "kitten" : ageYears < 10 ? "adult" : "senior";

  return { ageLabel, ageYears, ageGroup, sourceCatToken, publishedCatId };
}

function parseWav(buffer, filename) {
  if (buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error(`Not a RIFF/WAVE file: ${filename}`);
  }

  let offset = 12;
  let format;
  let dataBytes;
  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const chunkStart = offset + 8;
    if (chunkId === "fmt " && chunkSize >= 16) {
      format = {
        audioFormat: buffer.readUInt16LE(chunkStart),
        channels: buffer.readUInt16LE(chunkStart + 2),
        sampleRateHz: buffer.readUInt32LE(chunkStart + 4),
        byteRate: buffer.readUInt32LE(chunkStart + 8),
        blockAlign: buffer.readUInt16LE(chunkStart + 12),
        bitsPerSample: buffer.readUInt16LE(chunkStart + 14),
      };
    } else if (chunkId === "data") {
      dataBytes = Math.min(chunkSize, Math.max(0, buffer.length - chunkStart));
    }
    offset = chunkStart + chunkSize + (chunkSize % 2);
  }

  if (!format || dataBytes === undefined || !format.byteRate) {
    throw new Error(`Missing WAV fmt/data metadata: ${filename}`);
  }
  return {
    ...format,
    dataBytes,
    durationSeconds: dataBytes / format.byteRate,
  };
}

function buildEmbeddingMetadata(rows) {
  const byCat = new Map();
  for (const row of rows) {
    const current = byCat.get(row.cat_id) ?? {
      catId: row.cat_id,
      genders: new Set(),
      targetYears: new Set(),
      rows: 0,
    };
    current.genders.add(row.gender);
    current.targetYears.add(Number(row.target));
    current.rows += 1;
    byCat.set(row.cat_id, current);
  }
  for (const [catId, metadata] of byCat) {
    if (metadata.genders.size !== 1) {
      throw new Error(`Inconsistent embedding gender labels for cat_id ${catId}`);
    }
  }
  return byCat;
}

async function locateEmbeddingCsv(explicitPath, verifyOnly) {
  const canonicalPath = path.join(
    repoRoot,
    "data",
    "meowagenet",
    `official-${SOURCE_COMMIT.slice(0, 12)}`,
    "embeddings",
    "vggish_looped_embeddings.csv",
  );
  const candidates = [
    explicitPath,
    canonicalPath,
    path.join(
      repoRoot,
      "src",
      "baselines",
      "feline-age-prediction",
      "dataset",
      "embeddings",
      "vggish_looped_embeddings.csv",
    ),
  ].filter(Boolean);

  for (const candidate of candidates) {
    try {
      await readFile(candidate);
      return candidate;
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
  if (verifyOnly) {
    throw new Error("Could not locate vggish_looped_embeddings.csv; pass --embedding-csv PATH");
  }
  const blobUrl = `https://api.github.com/repos/aster-droide/feline-age-prediction/git/blobs/${EXPECTED_EMBEDDING_BLOB}`;
  const blob = await fetchWithRetry(blobUrl, "json");
  const buffer = Buffer.from(blob.content.replaceAll("\n", ""), blob.encoding);
  if (gitBlobHash(buffer) !== EXPECTED_EMBEDDING_BLOB) {
    throw new Error("Downloaded VGGish embedding CSV failed Git blob verification");
  }
  await mkdir(path.dirname(canonicalPath), { recursive: true });
  await writeFile(canonicalPath, buffer);
  return canonicalPath;
}

async function mapConcurrent(items, concurrency, worker) {
  const results = new Array(items.length);
  let nextIndex = 0;
  async function runner() {
    while (true) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= items.length) return;
      results[index] = await worker(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: concurrency }, () => runner()));
  return results;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const embeddingCsvPath = await locateEmbeddingCsv(args.embeddingCsv, args.verifyOnly);
  const embeddingBuffer = await readFile(embeddingCsvPath);
  const embeddingBlob = gitBlobHash(embeddingBuffer);
  if (embeddingBlob !== EXPECTED_EMBEDDING_BLOB) {
    throw new Error(
      `Unexpected VGGish CSV Git blob ${embeddingBlob}; expected ${EXPECTED_EMBEDDING_BLOB}`,
    );
  }
  const embeddingRows = parseSimpleCsv(embeddingBuffer.toString("utf8"));
  const embeddingByCat = buildEmbeddingMetadata(embeddingRows);

  const treeUrl = `https://api.github.com/repos/aster-droide/feline-age-prediction/git/trees/${SOURCE_COMMIT}?recursive=1`;
  const tree = await fetchWithRetry(treeUrl, "json");
  if (tree.truncated) throw new Error("GitHub tree response was truncated");
  const sourceFiles = tree.tree
    .filter((entry) => entry.type === "blob" && entry.path.startsWith(SOURCE_PREFIX))
    .sort((left, right) => left.path.localeCompare(right.path));
  if (sourceFiles.length !== 793) {
    throw new Error(`Expected 793 AudioCropped files, found ${sourceFiles.length}`);
  }

  await mkdir(rawRoot, { recursive: true });
  await mkdir(metadataRoot, { recursive: true });
  let completed = 0;
  const fileRows = await mapConcurrent(sourceFiles, CONCURRENCY, async (entry) => {
    const filename = path.posix.basename(entry.path);
    const localPath = path.join(rawRoot, filename);
    let buffer;
    try {
      buffer = await readFile(localPath);
      if (buffer.length !== entry.size || gitBlobHash(buffer) !== entry.sha) {
        if (args.verifyOnly) {
          throw new Error(`Local file failed verification: ${localPath}`);
        }
        buffer = undefined;
      }
    } catch (error) {
      if (error.code !== "ENOENT" && buffer !== undefined) throw error;
      if (args.verifyOnly) throw new Error(`Missing local file: ${localPath}`);
    }

    if (!buffer) {
      const rawUrl = `https://raw.githubusercontent.com/aster-droide/feline-age-prediction/${SOURCE_COMMIT}/${entry.path}`;
      buffer = await fetchWithRetry(rawUrl);
      if (buffer.length !== entry.size || gitBlobHash(buffer) !== entry.sha) {
        throw new Error(`Downloaded file failed Git blob verification: ${entry.path}`);
      }
      await writeFile(localPath, buffer);
    }

    const parsed = parseFilename(filename);
    const embedding = embeddingByCat.get(parsed.publishedCatId);
    if (!embedding) {
      throw new Error(`No embedding metadata for ${filename} (${parsed.publishedCatId})`);
    }
    const publishedTargetYears = Number(parsed.ageLabel.match(/^(\d+(?:\.\d+)?)Y/)[1]);
    if (!embedding.targetYears.has(publishedTargetYears)) {
      throw new Error(
        `Filename target ${publishedTargetYears} is absent from embedding labels for ${filename}`,
      );
    }
    const wav = parseWav(buffer, filename);
    completed += 1;
    if (completed % 50 === 0 || completed === sourceFiles.length) {
      process.stdout.write(`Verified ${completed}/${sourceFiles.length}\n`);
    }
    return {
      source_path: entry.path,
      local_relpath: path.relative(repoRoot, localPath).split(path.sep).join("/"),
      filename,
      source_git_blob_sha1: entry.sha,
      sha256: hash("sha256", buffer),
      size_bytes: buffer.length,
      sample_rate_hz: wav.sampleRateHz,
      channels: wav.channels,
      bits_per_sample: wav.bitsPerSample,
      duration_seconds: wav.durationSeconds.toFixed(6),
      age_label: parsed.ageLabel,
      age_years_filename: parsed.ageYears.toFixed(6),
      age_group_filename: parsed.ageGroup,
      source_cat_token: parsed.sourceCatToken,
      published_cat_id: parsed.publishedCatId,
      gender_published: [...embedding.genders][0],
      target_years_published: publishedTargetYears,
      duplicate_of: "",
      analysis_include: true,
      analysis_cat_id: parsed.publishedCatId,
      qc_flags: [
        parsed.sourceCatToken !== parsed.publishedCatId ? "cat_token_020AB_mapped_to_020A" : "",
        embedding.targetYears.size > 1 ? "same_cat_multiple_target_years" : "",
      ]
        .filter(Boolean)
        .join(";"),
    };
  });

  const rowsBySha = new Map();
  for (const row of fileRows) {
    const group = rowsBySha.get(row.sha256) ?? [];
    group.push(row);
    rowsBySha.set(row.sha256, group);
  }
  const duplicateGroups = [];
  for (const group of rowsBySha.values()) {
    if (group.length < 2) continue;
    const canonical = group[0];
    duplicateGroups.push(group.map((row) => row.source_path));
    for (const duplicate of group.slice(1)) {
      duplicate.duplicate_of = canonical.source_path;
      duplicate.analysis_include = false;
      duplicate.analysis_cat_id = canonical.published_cat_id;
      duplicate.qc_flags = [duplicate.qc_flags, "exact_audio_duplicate_excluded_from_clean_analysis"]
        .filter(Boolean)
        .join(";");
    }
    canonical.qc_flags = [canonical.qc_flags, "exact_audio_duplicate_canonical_copy"]
      .filter(Boolean)
      .join(";");
  }

  const fileColumns = [
    "source_path",
    "local_relpath",
    "filename",
    "source_git_blob_sha1",
    "sha256",
    "size_bytes",
    "sample_rate_hz",
    "channels",
    "bits_per_sample",
    "duration_seconds",
    "age_label",
    "age_years_filename",
    "age_group_filename",
    "source_cat_token",
    "published_cat_id",
    "gender_published",
    "target_years_published",
    "duplicate_of",
    "analysis_include",
    "analysis_cat_id",
    "qc_flags",
  ];
  const manifestText = toCsv(fileRows, fileColumns);
  const manifestPath = path.join(metadataRoot, "data_manifest.csv");
  await writeFile(manifestPath, manifestText, "utf8");

  const checksumText = fileRows
    .map((row) => `${row.sha256}  ${row.local_relpath}`)
    .join("\n") + "\n";
  const checksumPath = path.join(metadataRoot, "checksums.sha256");
  await writeFile(checksumPath, checksumText, "utf8");

  const catRows = [];
  for (const catId of [...embeddingByCat.keys()].sort()) {
    const matchingFiles = fileRows.filter((row) => row.published_cat_id === catId);
    if (matchingFiles.length === 0) {
      throw new Error(`Embedding cat_id ${catId} has no raw files`);
    }
    const embedding = embeddingByCat.get(catId);
    const targetYears = [...embedding.targetYears].sort((left, right) => left - right);
    const ageGroups = [
      ...new Set(targetYears.map((age) => (age < 0.5 ? "kitten" : age < 10 ? "adult" : "senior"))),
    ];
    const analysisIncluded = matchingFiles.some((row) => row.analysis_include);
    catRows.push({
      published_cat_id: catId,
      analysis_cat_id: analysisIncluded ? catId : matchingFiles[0].analysis_cat_id,
      source_cat_tokens: [...new Set(matchingFiles.map((row) => row.source_cat_token))].join(";"),
      gender_published: [...embedding.genders][0],
      target_years_published: targetYears.join(";"),
      age_group_published: ageGroups.join(";"),
      raw_file_rows: matchingFiles.length,
      unique_audio_rows: new Set(matchingFiles.map((row) => row.sha256)).size,
      vggish_embedding_rows: embedding.rows,
      embedding_surplus_rows: embedding.rows - matchingFiles.length,
      analysis_include: analysisIncluded,
      qc_flags: [...new Set(matchingFiles.flatMap((row) => row.qc_flags.split(";")).filter(Boolean))].join(
        ";",
      ),
    });
  }
  const catColumns = [
    "published_cat_id",
    "analysis_cat_id",
    "source_cat_tokens",
    "gender_published",
    "target_years_published",
    "age_group_published",
    "raw_file_rows",
    "unique_audio_rows",
    "vggish_embedding_rows",
    "embedding_surplus_rows",
    "analysis_include",
    "qc_flags",
  ];
  const catManifestText = toCsv(catRows, catColumns);
  const catManifestPath = path.join(metadataRoot, "cat_id_manifest.csv");
  await writeFile(catManifestPath, catManifestText, "utf8");

  const sourceTree = tree.tree.find((entry) => entry.path === "dataset/raw_audio/AudioCropped");
  const summary = {
    audit_date: "2026-08-25",
    source_repository: SOURCE_REPOSITORY,
    source_commit: SOURCE_COMMIT,
    source_tree_path: "dataset/raw_audio/AudioCropped",
    source_tree_git_sha1: sourceTree?.sha ?? null,
    official_file_rows: fileRows.length,
    official_total_bytes: fileRows.reduce((total, row) => total + row.size_bytes, 0),
    unique_audio_contents: rowsBySha.size,
    source_cat_tokens: new Set(fileRows.map((row) => row.source_cat_token)).size,
    published_cat_ids: embeddingByCat.size,
    clean_analysis_file_rows: fileRows.filter((row) => row.analysis_include).length,
    clean_analysis_cat_ids: new Set(
      fileRows.filter((row) => row.analysis_include).map((row) => row.analysis_cat_id),
    ).size,
    clean_analysis_embedding_rows: catRows
      .filter((row) => row.analysis_include)
      .reduce((total, row) => total + row.vggish_embedding_rows, 0),
    exact_duplicate_groups: duplicateGroups,
    manifest_sha256: hash("sha256", Buffer.from(manifestText, "utf8")),
    checksum_list_sha256: hash("sha256", Buffer.from(checksumText, "utf8")),
    embedding_csv: {
      local_source: path.relative(repoRoot, embeddingCsvPath).split(path.sep).join("/"),
      source_git_blob_sha1: embeddingBlob,
      sha256: hash("sha256", embeddingBuffer),
      rows: embeddingRows.length,
      dimensions: 128,
      published_cat_ids: embeddingByCat.size,
    },
    interpretation: {
      official_dataset_unit: "cropped_call_file",
      embedding_row_unit: "VGGish window embedding; one call can produce more than one row",
      clean_analysis_rule:
        "Keep the official 793-row manifest immutable; exclude later copies within exact SHA-256 duplicate groups from clean model training/evaluation.",
    },
  };
  const summaryPath = path.join(metadataRoot, "dataset_summary.json");
  await writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");

  process.stdout.write(`Manifest: ${manifestPath}\n`);
  process.stdout.write(`Checksums: ${checksumPath}\n`);
  process.stdout.write(`Cat IDs: ${catManifestPath}\n`);
  process.stdout.write(`Summary: ${summaryPath}\n`);
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error}\n`);
  process.exitCode = 1;
});
