"""Load-time selection and manifest verification for the offline fp8 prequant artifact.

A prequant artifact holds the ``torch.float8_e4m3fn`` tensors that
``fp8_cast.TRANSFORMER_LINEAR_DOWNCAST_MAP`` derives from the bf16 source at load
time. It is stored under the source checkpoint's own key names, carries the
source ``__metadata__`` and the prequant provenance entries, so a loader pointed
at it runs the identical policy over identical bytes: the loaded state dict is
unchanged, and the bf16 read plus the bf16->fp8 cast leave the load path.

``manifest.json`` beside the artifact is the contract. Every selection is verified
against it before the artifact is used:

* no artifact for the requested source -> online quantization, WARNING
* manifest or header missing/unreadable, artifact name, source name, source size,
  header/manifest disagreement, or key-count disagreement -> ``ProvenanceError``
* recorded quantization policy or dtype differs from the running program's, or
  the producer md5 map of the running program differs from the recorded one ->
  online quantization, WARNING naming the drift and the regenerate command

``LTX_PREQUANT`` selects the artifact (``1`` by default, ``0`` forces the online
path, any other value raises). ``LTX_ORIGINAL_PATH=1`` forces the online path.
``LTX_PREQUANT_VERIFY_SHA256=1`` additionally re-hashes the artifact and the
source checkpoint at load time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from safetensors import safe_open

ONLINE_MODE = "online-quant"
PREQUANT_MODE = "prequant"

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = "ltx-prequant-manifest/1"
QUANTIZED_DTYPE = "torch.float8_e4m3fn"
POLICY_ID = "ltx_core.quantization.fp8_cast.build_policy"
QUANTIZATION_FORMAT = "fp8-e4m3fn-cast"
PROJECT_DIR = os.environ.get("LTX_PROJECT_DIR", "/home/simon/ltx2.5-demo-main")

# The files whose contents decide which tensors the load-time policy emits.
PRODUCER_FILES = (
    "LTX-2/packages/ltx-core/src/ltx_core/quantization/fp8_cast.py",
    "LTX-2/packages/ltx-core/src/ltx_core/loader/sd_ops.py",
    "LTX-2/packages/ltx-core/src/ltx_core/loader/sft_loader.py",
    "LTX-2/packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py",
    "LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py",
    "LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py",
)

_REQUIRED_MANIFEST_KEYS = ("schema", "artifact", "artifact_sha256", "source_checkpoint", "quantization", "counts", "producer")
_REQUIRED_HEADER_KEYS = (
    "config",
    "model_version",
    "prequant_format_version",
    "prequant_dtype",
    "prequant_policy",
    "prequant_source_name",
    "prequant_source_size",
    "prequant_source_sha256",
    "prequant_producer_md5",
    "prequant_tool_version",
)
_READ_CHUNK = 8 << 20


class ProvenanceError(ValueError):
    """The artifact does not provably belong to the requested source checkpoint."""


def prequant_enabled() -> bool:
    """Whether the prequant artifact is selected (``LTX_PREQUANT``, default ``1``)."""
    if os.environ.get("LTX_ORIGINAL_PATH", "0") == "1":
        return False
    value = os.environ.get("LTX_PREQUANT", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_PREQUANT must be 0 or 1")
    return value == "1"


def deep_verify_enabled() -> bool:
    """Whether ``LTX_PREQUANT_VERIFY_SHA256=1`` re-hashes both files at load."""
    value = os.environ.get("LTX_PREQUANT_VERIFY_SHA256", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_PREQUANT_VERIFY_SHA256 must be 0 or 1")
    return value == "1"


def artifact_name_for(source_name: str) -> str:
    """Return the artifact filename the generator writes for *source_name*."""
    return source_name.removesuffix(".safetensors") + "-fp8e4m3fn.safetensors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while chunk := f.read(_READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def producer_map(project_dir: str | Path = PROJECT_DIR) -> dict[str, str]:
    """md5 of every file whose content defines the load-time policy's output."""
    root = Path(project_dir)
    return {rel: hashlib.md5((root / rel).read_bytes()).hexdigest() for rel in PRODUCER_FILES}


def _read_json(path: Path, what: str) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        raise ProvenanceError(f"{what} is unreadable: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"{what} is not valid JSON: {path}: {exc}") from exc


def _read_header(path: Path) -> tuple[dict[str, str], int]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return dict(f.metadata() or {}), len(f.keys())
    except Exception as exc:  # noqa: BLE001
        raise ProvenanceError(f"prequant artifact header is unreadable: {path}: {exc}") from exc


def check_provenance(manifest: dict, header: dict[str, str], key_count: int, source: Path, artifact: Path) -> None:
    """Raise :class:`ProvenanceError` unless the artifact provably serves *source*."""
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ProvenanceError(f"prequant manifest {artifact.parent / MANIFEST_NAME} is missing keys {missing}")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ProvenanceError(f"prequant manifest schema {manifest['schema']!r} is not {MANIFEST_SCHEMA!r}")
    if manifest["artifact"] != artifact.name:
        raise ProvenanceError(f"prequant manifest names artifact {manifest['artifact']!r}, but {artifact.name!r} was found")
    if int(manifest["artifact_bytes"]) != artifact.stat().st_size:
        raise ProvenanceError(
            f"prequant artifact {artifact} is {artifact.stat().st_size} bytes, manifest says {manifest['artifact_bytes']}"
        )
    missing_header = [key for key in _REQUIRED_HEADER_KEYS if key not in header]
    if missing_header:
        raise ProvenanceError(f"prequant artifact {artifact} is missing header entries {missing_header}")

    src = manifest["source_checkpoint"]
    if src["name"] != source.name:
        raise ProvenanceError(
            f"prequant artifact {artifact.name} was generated from {src['name']!r}, but the pipeline is loading {source.name!r}"
        )
    if int(src["bytes"]) != source.stat().st_size:
        raise ProvenanceError(
            f"prequant artifact {artifact.name} records source size {src['bytes']}, but {source} is {source.stat().st_size} bytes"
        )
    if header["prequant_source_name"] != src["name"] or header["prequant_source_sha256"] != src["sha256"]:
        raise ProvenanceError(
            f"prequant artifact {artifact} header and manifest disagree about the source checkpoint "
            f"({header['prequant_source_name']} / {header['prequant_source_sha256'][:12]} vs {src['name']} / {src['sha256'][:12]})"
        )
    if int(manifest["counts"]["output_keys"]) != key_count:
        raise ProvenanceError(
            f"prequant artifact {artifact} holds {key_count} tensors, manifest says {manifest['counts']['output_keys']}"
        )


def format_drift(manifest: dict) -> list[str]:
    """Return the reasons the artifact's format or producer differs from this program."""
    quant = manifest.get("quantization", {})
    drift = []
    if quant.get("format") != QUANTIZATION_FORMAT:
        drift.append(f"format {quant.get('format')!r} != {QUANTIZATION_FORMAT!r}")
    if quant.get("quantized_dtype") != QUANTIZED_DTYPE:
        drift.append(f"quantized_dtype {quant.get('quantized_dtype')!r} != {QUANTIZED_DTYPE!r}")
    if quant.get("policy") != POLICY_ID:
        drift.append(f"policy {quant.get('policy')!r} != {POLICY_ID!r}")
    recorded = manifest.get("producer", {}).get("md5", {})
    running = producer_map()
    for rel, digest in running.items():
        if rel not in recorded:
            drift.append(f"producer file {rel} is not recorded in the manifest")
        elif recorded[rel] != digest:
            drift.append(f"producer file {rel} md5 {recorded[rel][:8]} != running {digest[:8]}")
    for rel in recorded:
        if rel not in running:
            drift.append(f"producer file {rel} is recorded but not part of the running program's set")
    return drift


def resolve_transformer_checkpoint(
    source: str | Path,
    *,
    models_dir: str | Path,
    log: logging.Logger,
) -> tuple[str, str]:
    """Return ``(transformer_checkpoint_path, mode)`` for the transformer weights.

    Returns the verified prequant artifact with mode ``"prequant"``, or *source*
    with mode ``"online-quant"`` when the artifact is absent or was built by a
    different format/producer. A manifest that does not describe *source*, or an
    artifact whose header disagrees with it, raises :class:`ProvenanceError`.
    """
    source_path = Path(source).resolve()
    if not prequant_enabled():
        log.info("prequant disabled: loading %s with online bf16->fp8 quantization", source_path.name)
        return str(source_path), ONLINE_MODE

    prequant_dir = Path(os.environ.get("LTX_PREQUANT_DIR", Path(models_dir) / "prequant"))
    artifact = prequant_dir / artifact_name_for(source_path.name)
    if not artifact.exists():
        log.warning(
            "no prequant artifact at %s; loading %s with online bf16->fp8 quantization",
            artifact,
            source_path.name,
        )
        return str(source_path), ONLINE_MODE

    manifest = _read_json(prequant_dir / MANIFEST_NAME, "prequant manifest")
    header, key_count = _read_header(artifact)
    check_provenance(manifest, header, key_count, source_path, artifact)

    drift = format_drift(manifest)
    if drift:
        log.warning(
            "prequant artifact %s was not built by this program version (%s); "
            "loading %s with online bf16->fp8 quantization. Regenerate with: %s",
            artifact,
            "; ".join(drift),
            source_path.name,
            manifest.get("regenerate", {}).get("command", "make_prequant.py generate"),
        )
        return str(source_path), ONLINE_MODE

    if deep_verify_enabled():
        artifact_sha = sha256_file(artifact)
        if artifact_sha != manifest["artifact_sha256"]:
            raise ProvenanceError(
                f"prequant artifact {artifact} sha256 {artifact_sha} != manifest {manifest['artifact_sha256']}"
            )
        source_sha = sha256_file(source_path)
        if source_sha != manifest["source_checkpoint"]["sha256"]:
            raise ProvenanceError(
                f"source checkpoint {source_path} sha256 {source_sha} != manifest {manifest['source_checkpoint']['sha256']}"
            )

    log.info(
        "loading prequant fp8 artifact %s (%d bytes, %d tensors, source sha256 %s, producer digest %s, tool v%s); "
        "online bf16->fp8 quantization is skipped",
        artifact,
        artifact.stat().st_size,
        key_count,
        manifest["source_checkpoint"]["sha256"],
        manifest["producer"].get("digest", "unrecorded")[:12],
        header["prequant_tool_version"],
    )
    return str(artifact), PREQUANT_MODE
