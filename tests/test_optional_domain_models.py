import csv

import pandas as pd
import pytest

from src import optional_domain_models as odm
import src.biomedical_evidence as biomedical_evidence_module
import src.patent_evidence_embeddings as patent_evidence_module
from src.biomedical_evidence import OUTPUT_COLUMNS as BIOMEDICAL_OUTPUT_COLUMNS
from src.patent_evidence_embeddings import OUTPUT_COLUMNS as PATENT_OUTPUT_COLUMNS


class FakeEncoder:
    embedding_backend = "fake_backend"
    pooling_method = "fake_pooling"
    model_source = "fake/model"

    def encode(self, sentences, **kwargs):
        return [[1.0, 0.0, 0.0] if index == 0 else [0.0, 1.0, 0.0] for index, _ in enumerate(sentences)]


def test_allow_local_model_downloads_missing_disables_large_model_loading(monkeypatch):
    monkeypatch.delenv(odm.ALLOW_LOCAL_MODEL_DOWNLOADS_ENV, raising=False)
    selection = odm.resolve_model_selection(
        model_type="biomedical",
        option="BioBERT",
        fallback_model_id="sentence-transformers/all-MiniLM-L6-v2",
    )

    with pytest.raises(odm.DomainModelUnavailableError) as excinfo:
        odm.load_optional_model(selection)

    assert excinfo.value.status == "downloads_disabled"


def test_failed_model_loading_returns_benchmark_status_without_crash(monkeypatch):
    monkeypatch.delenv(odm.ALLOW_LOCAL_MODEL_DOWNLOADS_ENV, raising=False)
    selection = odm.resolve_model_selection(
        model_type="patent",
        option="PatentSBERTa",
        fallback_model_id="sentence-transformers/all-MiniLM-L6-v2",
    )

    rows = odm.benchmark_rows(
        [selection],
        query_texts=("query",),
        evidence_texts=("evidence",),
    )

    assert rows[0]["model_status"] == "downloads_disabled"
    assert "ALLOW_LOCAL_MODEL_DOWNLOADS=1" in rows[0]["error_message"]


def test_transformers_mean_pooling_wrapper_records_metadata():
    encoder = object.__new__(odm.TransformersMeanPoolingEncoder)
    encoder.model_source = "dmis-lab/biobert-base-cased-v1.1"

    assert encoder.embedding_backend == "transformers"
    assert encoder.pooling_method == "mean_pooling"
    assert odm.encoder_metadata(encoder) == {
        "embedding_backend": "transformers",
        "pooling_method": "mean_pooling",
        "model_source": "dmis-lab/biobert-base-cased-v1.1",
    }


def test_benchmark_optional_models_writes_required_schema(tmp_path):
    biomedical = odm.resolve_model_selection(
        model_type="biomedical",
        option=odm.CLOUD_SAFE_FALLBACK_LABEL,
        fallback_model_id="fallback-biomedical",
    )
    patent = odm.resolve_model_selection(
        model_type="patent",
        option=odm.CLOUD_SAFE_FALLBACK_LABEL,
        fallback_model_id="fallback-patent",
    )

    result = odm.benchmark_optional_models(
        biomedical_selection=biomedical,
        patent_selection=patent,
        output_dir=tmp_path,
        loader=lambda selection: FakeEncoder(),
    )

    for path in (result["biomedical_path"], result["patent_path"]):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == odm.BENCHMARK_COLUMNS
            rows = list(reader)
            assert rows
            assert rows[0]["model_status"] == "available"
            assert rows[0]["embedding_dimension"] == "3"


def test_step_six_and_step_seven_required_columns_are_preserved():
    biomedical_required = (
        "molecule_id",
        "biomedical_model_name",
        "biomedical_model_status",
        "biomedical_evidence_status",
        "biomedical_similarity_score",
        "biomedical_relevance_category",
        "biomedical_evidence_count",
        "top_biomedical_evidence_id",
        "top_biomedical_evidence_text",
        "evidence_note",
    )
    patent_required = (
        "molecule_id",
        "patent_model_name",
        "patent_model_status",
        "patent_evidence_status",
        "patent_similarity_score",
        "patent_relevance_category",
        "patent_evidence_count",
        "top_patent_evidence_id",
        "top_patent_evidence_text",
        "surechembl_structure_status",
        "patent_document_metadata_status",
        "evidence_note",
    )

    assert BIOMEDICAL_OUTPUT_COLUMNS[: len(biomedical_required)] == biomedical_required
    assert PATENT_OUTPUT_COLUMNS[: len(patent_required)] == patent_required
    assert BIOMEDICAL_OUTPUT_COLUMNS[-6:] == (
        "embedding_backend",
        "pooling_method",
        "model_source",
        "preferred_model_name",
        "fallback_model_name",
        "actual_model_used",
    )
    assert PATENT_OUTPUT_COLUMNS[-6:] == (
        "embedding_backend",
        "pooling_method",
        "model_source",
        "preferred_model_name",
        "fallback_model_name",
        "actual_model_used",
    )


def test_paecter_requires_custom_model_id_note():
    selection = odm.resolve_model_selection(
        model_type="patent",
        option="PaECTER",
        fallback_model_id="sentence-transformers/all-MiniLM-L6-v2",
    )

    assert selection.model_id == ""
    assert "not verified" in selection.error_message


def test_downloads_disabled_biomedical_csv_does_not_load_model(monkeypatch, tmp_path):
    context = tmp_path / "compound_context.csv"
    evidence = tmp_path / "text_evidence.csv"
    output = tmp_path / "biomedical_evidence.csv"
    context.write_text("molecule_id,compound_description\nmol_a,kinase context\n", encoding="utf-8")
    evidence.write_text("evidence_id,molecule_id,source_type,title,text\nev_1,,local,Demo,kinase evidence\n", encoding="utf-8")
    monkeypatch.setattr(
        biomedical_evidence_module,
        "load_model",
        lambda *args, **kwargs: pytest.fail("model loader should not run"),
    )

    biomedical_evidence_module.biomedical_evidence_csv(
        context,
        evidence,
        output,
        model_name="dmis-lab/biobert-base-cased-v1.1",
        unavailable_status="embeddings_disabled",
        unavailable_metadata={
            "embedding_backend": "transformers",
            "pooling_method": "mean_pooling",
            "model_source": "dmis-lab/biobert-base-cased-v1.1",
            "preferred_model_name": "dmis-lab/biobert-base-cased-v1.1",
            "fallback_model_name": odm.FALLBACK_MODEL_ID,
            "actual_model_used": "",
        },
    )

    frame = pd.read_csv(output)
    assert frame.loc[0, "biomedical_model_status"] == "embeddings_disabled"
    assert frame.loc[0, "embedding_backend"] == "transformers"
    assert frame.loc[0, "pooling_method"] == "mean_pooling"
    assert frame.loc[0, "model_source"] == "dmis-lab/biobert-base-cased-v1.1"
    assert frame.loc[0, "preferred_model_name"] == "dmis-lab/biobert-base-cased-v1.1"
    assert frame.loc[0, "fallback_model_name"] == odm.FALLBACK_MODEL_ID


def test_downloads_disabled_patent_csv_does_not_load_model(monkeypatch, tmp_path):
    surechembl = tmp_path / "surechembl_evidence.csv"
    output = tmp_path / "patent_evidence_embeddings.csv"
    surechembl.write_text(
        "molecule_id,lookup_status,patent_title,evidence_note\n"
        "mol_a,match_found,Kinase patent,SureChEMBL evidence\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        patent_evidence_module,
        "load_model",
        lambda *args, **kwargs: pytest.fail("model loader should not run"),
    )

    patent_evidence_module.patent_evidence_embeddings_csv(
        surechembl,
        output,
        model_name="AI-Growth-Lab/PatentSBERTa",
        unavailable_status="embeddings_disabled",
        unavailable_metadata={
            "embedding_backend": "sentence-transformers",
            "pooling_method": "model_default",
            "model_source": "AI-Growth-Lab/PatentSBERTa",
            "preferred_model_name": "AI-Growth-Lab/PatentSBERTa",
            "fallback_model_name": odm.FALLBACK_MODEL_ID,
            "actual_model_used": "",
        },
    )

    frame = pd.read_csv(output)
    assert frame.loc[0, "patent_model_status"] == "embeddings_disabled"
    assert frame.loc[0, "embedding_backend"] == "sentence-transformers"
    assert frame.loc[0, "pooling_method"] == "model_default"
    assert frame.loc[0, "model_source"] == "AI-Growth-Lab/PatentSBERTa"
    assert frame.loc[0, "preferred_model_name"] == "AI-Growth-Lab/PatentSBERTa"
    assert frame.loc[0, "fallback_model_name"] == odm.FALLBACK_MODEL_ID