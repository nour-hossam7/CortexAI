"""End-to-end verification script for the CortexAI NLP module."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp_module import (  # noqa: E402
    NLPConfig,
    build_config,
    configure_logging,
    extract_features_for_fusion,
    get_feature_dimension,
    load_embeddings,
    load_features_for_fusion,
    load_labels,
    load_metadata,
    load_textbrats_dataset,
    preprocess_reports,
)
from src.nlp_module.data_loader import inspect_textbrats_dataset
from src.nlp_module.embeddings import TransformerEmbedder
from src.nlp_module.feature_extractor import NLPFeatureExtractor
from src.nlp_module.pipeline import NLPPipeline
from src.nlp_module.preprocessing import PreprocessingOptions, TextPreprocessor
from src.nlp_module.tokenizer import TransformerTextTokenizer
from src.nlp_module.validator import (
    validate_reports_dataframe,
    validate_supported_files_readable,
)


class MockTokenizer:
    """Deterministic offline tokenizer for pipeline verification."""

    pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        max_length: int = 256,
        padding: bool | str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        input_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        for text in texts:
            tokens = [101] + [
                (abs(hash(word)) % 5000) + 100 for word in str(text).split()
            ][: max(0, max_length - 2)]
            tokens = tokens[:max_length]
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            if padding == "max_length":
                attention = [1] * len(tokens) + [0] * (max_length - len(tokens))
                tokens = tokens + [0] * (max_length - len(tokens))
            else:
                attention = [1] * len(tokens)
            input_rows.append(torch.tensor(tokens, dtype=torch.long))
            mask_rows.append(torch.tensor(attention, dtype=torch.long))
        return {
            "input_ids": torch.stack(input_rows),
            "attention_mask": torch.stack(mask_rows),
        }


class MockModel:
    """Deterministic offline transformer for embedding verification."""

    hidden_size = 32

    def eval(self) -> "MockModel":
        return self

    def to(self, device: str) -> "MockModel":
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> Any:
        batch_size, seq_len = input_ids.shape
        base = input_ids.float().mean(dim=1, keepdim=True)
        token_vectors = base.unsqueeze(1).expand(batch_size, seq_len, self.hidden_size)
        token_vectors = token_vectors + torch.arange(self.hidden_size).float() * 1e-4
        return type("ModelOutput", (), {"last_hidden_state": token_vectors})()


def huggingface_models_available() -> bool:
    """Return True when BioBERT can be loaded from Hugging Face."""
    if hasattr(huggingface_models_available, '_cached'):
        return bool(huggingface_models_available._cached)  # type: ignore[attr-defined]
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        huggingface_models_available._cached = True  # type: ignore[attr-defined]
    except OSError:
        huggingface_models_available._cached = False  # type: ignore[attr-defined]
    return bool(huggingface_models_available._cached)  # type: ignore[attr-defined]


def build_test_extractor(config: NLPConfig) -> NLPFeatureExtractor:
    """Build a feature extractor using real or mock transformer components."""
    if huggingface_models_available():
        return NLPFeatureExtractor(config)
    mock_tokenizer = TransformerTextTokenizer(config, tokenizer=MockTokenizer())
    mock_embedder = TransformerEmbedder(config, model=MockModel())
    return NLPFeatureExtractor(
        config,
        tokenizer=mock_tokenizer,
        embedder=mock_embedder,
    )


def build_test_tokenizer(config: NLPConfig) -> TransformerTextTokenizer:
    """Build a tokenizer wrapper using real or mock components."""
    if huggingface_models_available():
        return TransformerTextTokenizer(config)
    return TransformerTextTokenizer(config, tokenizer=MockTokenizer())


def build_test_embedder(config: NLPConfig) -> TransformerEmbedder:
    """Build an embedder using real or mock components."""
    if huggingface_models_available():
        return TransformerEmbedder(config)
    return TransformerEmbedder(config, model=MockModel())


class VerificationFailure(Exception):
    """Raised when one verification check fails."""


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise VerificationFailure(f"{name} FAILED{': ' + detail if detail else ''}")
    print(f"[PASS] {name}")


def create_sample_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "reports.csv"
    csv_path.write_text(
        "report_id,text,diagnosis\n"
        "case_001,\"MRI brain: 45 y/o pt w/ enhancing lesion in L temporal lobe. "
        "Findings suggest glioblastoma.\",glioblastoma\n"
        "case_002,\"Follow-up scan shows stable mass. No new enhancement.\",stable\n"
        "case_003,\"Post-operative changes s/p resection. Residual tumor suspected.\",residual\n",
        encoding="utf-8",
    )
    json_path = root / "extra_reports.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "id": "case_004",
                    "radiology_report": "Bilateral frontal edema with midline shift.",
                    "label": "edema",
                }
            ]
        ),
        encoding="utf-8",
    )
    txt_dir = root / "single_reports"
    txt_dir.mkdir(exist_ok=True)
    (txt_dir / "case_005.txt").write_text(
        "Impression: ring-enhancing lesion consistent with high-grade glioma.",
        encoding="utf-8",
    )


def verify_validation(raw_dir: Path, config: NLPConfig) -> None:
    inspection = inspect_textbrats_dataset(config)
    check("inspection finds supported files", len(inspection.supported_files) >= 2)

    partial_empty = pd.DataFrame(
        {"report_id": ["a", "b"], "text": ["valid report", ""]}
    )
    partial_report = validate_reports_dataframe(partial_empty)
    check(
        "validation warns on partial empty text",
        partial_report.empty_text_count == 1 and partial_report.is_valid,
    )

    all_empty = pd.DataFrame({"report_id": ["a"], "text": [""]})
    all_empty_report = validate_reports_dataframe(all_empty)
    check("validation errors when all text empty", not all_empty_report.is_valid)

    dup_df = pd.DataFrame({"report_id": ["x", "x"], "text": ["a", "b"]})
    dup_report = validate_reports_dataframe(dup_df)
    check("validation errors on duplicate IDs", dup_report.duplicate_id_count == 2)

    missing_id = pd.DataFrame({"report_id": ["", "b"], "text": ["a", "b"]})
    missing_id_report = validate_reports_dataframe(missing_id)
    check("validation errors on missing IDs", missing_id_report.missing_id_count == 1)

    cfg_with_labels = config.with_updates(label_columns=("diagnosis",))
    missing_labels = pd.DataFrame({"report_id": ["a"], "text": ["hello"]})
    label_report = validate_reports_dataframe(
        missing_labels,
        label_columns=("diagnosis",),
    )
    check("validation errors on missing label columns", not label_report.is_valid)

    missing_meta = pd.DataFrame({"report_id": ["a"], "text": ["hello"]})
    meta_report = validate_reports_dataframe(
        missing_meta,
        metadata_columns=("source_path",),
    )
    check("validation errors on missing metadata columns", not meta_report.is_valid)

    bad_root = raw_dir.parent / "validation_only_bad"
    bad_root.mkdir(exist_ok=True)
    bad_json = bad_root / "broken.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    bad_inspection = inspect_textbrats_dataset(
        config.with_updates(
            paths=config.paths.__class__(
                project_root=config.paths.project_root,
                raw_textbrats_dir=bad_root,
                processed_nlp_dir=config.paths.processed_nlp_dir,
                reports_dir=config.paths.reports_dir,
            )
        )
    )
    readable = validate_supported_files_readable(bad_inspection)
    check("validation detects corrupted JSON", readable.corrupted_file_count >= 1)


def verify_preprocessing(config: NLPConfig) -> None:
    raw = (
        "  MRI\u2014brain: 45 y/o pt w/ enhancing lesion.  "
        "Findings suggest glioblastoma (GBM).  "
    )
    options = PreprocessingOptions(
        lowercase=False,
        normalize_abbreviations=True,
    )
    cleaned = TextPreprocessor(options).clean_text(raw)
    check("preprocessing normalizes whitespace", "  " not in cleaned)
    check("preprocessing normalizes unicode dash", "\u2014" not in cleaned)
    check("preprocessing expands w/", " with " in cleaned)
    check("preprocessing preserves glioblastoma", "glioblastoma" in cleaned)
    check("preprocessing preserves GBM", "GBM" in cleaned)

    frame = pd.DataFrame({"report_id": ["a"], "text": [raw]})
    processed = preprocess_reports(frame)
    check("preprocessing adds clean_text column", "clean_text" in processed.columns)
    check(
        "preprocessing changes text",
        processed.loc[0, "clean_text"] != processed.loc[0, "text"],
    )


def verify_tokenization_and_embeddings(config: NLPConfig, reports: pd.DataFrame) -> None:
    processed = preprocess_reports(reports)
    tokenizer = build_test_tokenizer(config)
    tokenized = tokenizer.tokenize_dataframe(processed)
    max_len = config.max_length
    check("tokenized input_ids shape", tokenized.input_ids.shape[0] == len(reports))
    check(
        "tokenized sequence length respects max_length",
        tokenized.input_ids.shape[1] <= max_len,
    )
    check(
        "attention_mask matches input_ids shape",
        tokenized.attention_mask.shape == tokenized.input_ids.shape,
    )
    check(
        "attention_mask values are 0/1",
        set(tokenized.attention_mask.unique().tolist()).issubset({0, 1}),
    )

    for strategy in ("cls", "mean"):
        cfg = config.with_updates(pooling_strategy=strategy)
        embedder = build_test_embedder(cfg)
        result = embedder.embed_tokenized(tokenized)
        dim = result.embeddings.shape[1]
        check(f"{strategy} pooling produces 2D embeddings", result.embeddings.ndim == 2)
        check(f"{strategy} pooling dimension > 0", dim > 0)
        check(f"{strategy} pooling no NaN", not np.isnan(result.embeddings).any())
        check(f"{strategy} pooling no Inf", not np.isinf(result.embeddings).any())
        check(
            f"{strategy} pooling not all zeros",
            not np.allclose(result.embeddings, 0.0),
        )
        check(
            f"{strategy} pooling count matches reports",
            result.embeddings.shape[0] == len(reports),
        )

    identical = pd.DataFrame(
        {
            "report_id": ["id1", "id2"],
            "clean_text": ["same report text", "same report text"],
        }
    )
    tok_identical = build_test_tokenizer(config).tokenize_dataframe(identical)
    emb_identical = build_test_embedder(config).embed_tokenized(tok_identical)
    check(
        "identical reports produce identical embeddings",
        np.allclose(emb_identical.embeddings[0], emb_identical.embeddings[1]),
    )

    different = pd.DataFrame(
        {
            "report_id": ["id1", "id2"],
            "clean_text": [
                "glioblastoma with midline shift",
                "stable post-operative changes",
            ],
        }
    )
    tok_diff = build_test_tokenizer(config).tokenize_dataframe(different)
    emb_diff = build_test_embedder(config).embed_tokenized(tok_diff)
    check(
        "different reports produce different embeddings",
        not np.allclose(emb_diff.embeddings[0], emb_diff.embeddings[1]),
    )


def verify_saved_outputs(saved_dir: Path, config: NLPConfig, report_count: int) -> None:
    required = [
        "embeddings.npy",
        "labels.csv",
        "metadata.csv",
        "config.json",
        "nlp_embeddings.npz",
        "cleaned_reports.csv",
        "tokenized_inputs.pt",
        "nlp_eda_summary.json",
        "nlp_feature_manifest.json",
    ]
    for name in required:
        check(f"output file exists: {name}", (saved_dir / name).exists())

    embeddings = np.load(saved_dir / "embeddings.npy")
    labels = pd.read_csv(saved_dir / "labels.csv")
    metadata = pd.read_csv(saved_dir / "metadata.csv")
    check("embeddings shape rows", embeddings.shape[0] == report_count)
    check("labels count matches embeddings", len(labels) == report_count)
    check("metadata count matches embeddings", len(metadata) == report_count)
    check("embeddings dtype float32", embeddings.dtype == np.float32)

    loaded = load_embeddings(processed_dir=saved_dir)
    check("load_embeddings count", len(loaded.report_ids) == report_count)
    check(
        "load_embeddings values match",
        np.allclose(loaded.embeddings, embeddings),
    )

    dim = get_feature_dimension(processed_dir=saved_dir)
    check("get_feature_dimension", dim == embeddings.shape[1])

    labels_loaded = load_labels(processed_dir=saved_dir)
    metadata_loaded = load_metadata(processed_dir=saved_dir)
    check("load_labels", len(labels_loaded) == report_count)
    check("load_metadata", len(metadata_loaded) == report_count)

    fusion_frame = load_features_for_fusion(processed_dir=saved_dir)
    check("load_features_for_fusion rows", len(fusion_frame) == report_count)
    check(
        "load_features_for_fusion columns",
        fusion_frame.shape[1] == embeddings.shape[1] + 1,
    )


def verify_clinicalbert(config: NLPConfig, reports: pd.DataFrame) -> None:
    clinical_cfg = config.with_updates(model_alias="clinicalbert", pooling_strategy="mean")
    bundle = build_test_extractor(clinical_cfg).extract(reports)
    emb = bundle.embeddings.embeddings
    check("ClinicalBERT embeddings shape", emb.shape[0] == len(reports))
    check("ClinicalBERT no NaN", not np.isnan(emb).any())


def main() -> int:
    configure_logging()
    temp_root = Path(tempfile.mkdtemp(prefix="cortexai_nlp_verify_"))
    raw_dir = temp_root / "raw" / "textbrats"
    processed_dir = temp_root / "processed" / "nlp"
    create_sample_dataset(raw_dir)

    config = build_config(
        raw_textbrats_dir=raw_dir,
        processed_nlp_dir=processed_dir,
        batch_size=2,
        max_length=128,
    )
    using_mocks = not huggingface_models_available()
    if using_mocks:
        print(
            "[INFO] Hugging Face models unavailable in this environment; "
            "using deterministic mock transformer components for embedding tests."
        )

    failures: list[str] = []
    passed = 0

    def run_section(name: str, func) -> None:
        nonlocal passed
        print(f"\n=== {name} ===")
        try:
            func()
            passed += 1
        except VerificationFailure as exc:
            failures.append(f"{name}: {exc}")
            print(f"[FAIL] {exc}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"[ERROR] {exc}")
            traceback.print_exc()

    def section_validation() -> None:
        verify_validation(raw_dir, config)

    def section_preprocessing() -> None:
        verify_preprocessing(config)

    loaded_reports_holder: dict[str, pd.DataFrame] = {}

    def section_loading_capture() -> None:
        inspection = inspect_textbrats_dataset(config)
        check("dataset directory exists", inspection.exists)
        check("supported formats detected", ".csv" in inspection.files_by_extension)
        loaded = load_textbrats_dataset(config)
        check("reports loaded", len(loaded) >= 4)
        check("report_id column present", "report_id" in loaded.columns)
        check("text column present", "text" in loaded.columns)
        check("metadata columns present", "source_path" in loaded.columns)
        loaded_reports_holder["df"] = loaded

    def section_tokenization_embeddings() -> None:
        reports = loaded_reports_holder["df"]
        verify_tokenization_and_embeddings(config, reports)

    def section_pipeline() -> None:
        pipeline = NLPPipeline(
            config=config,
            feature_extractor=build_test_extractor(config),
        )
        result = pipeline.run()
        reports = loaded_reports_holder["df"]
        check("pipeline validation valid", result.validation.is_valid)
        check(
            "pipeline record_count not duplicated in manifest",
            result.validation.record_count == len(reports),
        )
        check(
            "pipeline embedding count",
            result.features.embeddings.embeddings.shape[0] == len(reports),
        )
        verify_saved_outputs(processed_dir, config, len(reports))

    def section_fusion_api() -> None:
        reports = loaded_reports_holder["df"]
        if using_mocks:
            with patch(
                "src.nlp_module.pipeline.NLPFeatureExtractor",
                side_effect=lambda cfg: build_test_extractor(cfg),
            ):
                fusion = extract_features_for_fusion(reports, config=config)
        else:
            fusion = extract_features_for_fusion(reports, config=config)
        check("extract_features_for_fusion rows", len(fusion) == len(reports))
        check(
            "extract_features_for_fusion has report_id column",
            "report_id" in fusion.columns,
        )

    def section_clinicalbert() -> None:
        reports = loaded_reports_holder["df"]
        verify_clinicalbert(config, reports)

    def section_inspect_only() -> None:
        inspect_result = NLPPipeline(config).inspect_dataset()
        check("inspect-only supported files", len(inspect_result.supported_files) >= 2)

    def section_real_dataset() -> None:
        real_dir = PROJECT_ROOT / "datasets" / "raw" / "textbrats"
        if not real_dir.exists():
            print("[SKIP] Real TextBraTS directory not present locally.")
            return
        real_cfg = build_config(raw_textbrats_dir=real_dir)
        real_inspection = inspect_textbrats_dataset(real_cfg)
        if real_inspection.is_empty:
            print("[SKIP] Real TextBraTS directory exists but contains no supported files.")
            return
        check("real dataset inspection", not real_inspection.is_empty)

    run_section("Validation", section_validation)
    run_section("Loading", section_loading_capture)
    run_section("Preprocessing", section_preprocessing)
    run_section("Tokenization & Embeddings", section_tokenization_embeddings)
    run_section("Full Pipeline", section_pipeline)
    run_section("Fusion API", section_fusion_api)
    run_section("ClinicalBERT", section_clinicalbert)
    run_section("Inspect Only", section_inspect_only)
    run_section("Real Dataset Check", section_real_dataset)

    shutil.rmtree(temp_root, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"VERIFICATION FAILED ({len(failures)} issue(s)):")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("ALL NLP MODULE VERIFICATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
