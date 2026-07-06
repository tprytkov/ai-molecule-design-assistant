import csv
from pathlib import Path

import pytest

import src.biomedical_evidence as biomedical_module
from src.biomedical_evidence import (
    BiomedicalModelUnavailableError,
    OUTPUT_COLUMNS,
    biomedical_evidence_csv,
)


class FakeBiomedicalEncoder:
    """Deterministic encoder for biomedical evidence tests."""

    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            text = str(sentence).lower()
            if "kinase" in text or "public comparator" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "unrelated" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.5, 0.5, 0.0])
        return vectors


def write_common_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    context_path = tmp_path / "compound_context.csv"
    evidence_path = tmp_path / "text_evidence.csv"
    identity_path = tmp_path / "chemical_identity.csv"
    descriptor_path = tmp_path / "descriptors.csv"
    output_path = tmp_path / "biomedical_evidence.csv"
    context_path.write_text(
        "molecule_id,closest_public_compound,reported_targets,"
        "biological_reference_summary,biomedical_relevance_summary\n"
        "mol_1,Public comparator,kinase,Kinase reference summary,"
        "Generated molecule kinase context.\n",
        encoding="utf-8",
    )
    evidence_path.write_text(
        "evidence_id,molecule_id,source_type,title,text\n"
        "ev_1,mol_1,local,Kinase evidence,Kinase signaling evidence.\n"
        "ev_2,mol_1,local,Unrelated evidence,Unrelated phenotype note.\n",
        encoding="utf-8",
    )
    identity_path.write_text(
        "molecule_id,exact_public_name,preferred_name\nmol_1,,Demo molecule\n",
        encoding="utf-8",
    )
    descriptor_path.write_text(
        "molecule_id,valid_smiles,druglikeness_category\nmol_1,True,favorable\n",
        encoding="utf-8",
    )
    return context_path, evidence_path, identity_path, descriptor_path, output_path


def test_biomedical_evidence_csv_writes_fallback_when_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context_path, evidence_path, identity_path, descriptor_path, output_path = (
        write_common_inputs(tmp_path)
    )
    monkeypatch.setattr(
        biomedical_module,
        "load_model",
        lambda model_name=biomedical_module.DEFAULT_BIOMEDICAL_MODEL: (
            _ for _ in ()
        ).throw(BiomedicalModelUnavailableError("missing model")),
    )

    count = biomedical_evidence_csv(
        context_path,
        evidence_path,
        output_path,
        identity_path=identity_path,
        descriptor_path=descriptor_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        rows = list(reader)
    assert count == 1
    assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS
    assert rows[0]["molecule_id"] == "mol_1"
    assert rows[0]["biomedical_model_status"] == "lexical_fallback_used"
    assert rows[0]["biomedical_evidence_status"] == "available"
    assert float(rows[0]["biomedical_similarity_score"]) > 0
    assert rows[0]["biomedical_relevance_category"] != "not_run"
    assert rows[0]["model_backend"] == "lexical_token_overlap"
    assert rows[0]["model_status"] == "lexical_fallback_used"
    assert rows[0]["model_cache_status"] == "not_required"
    assert rows[0]["fallback_used"] == "yes"
    assert "Lexical fallback text-similarity triage only" in rows[0]["evidence_note"]


def test_biomedical_evidence_csv_scores_with_fake_encoder(
    tmp_path: Path,
) -> None:
    context_path, evidence_path, identity_path, descriptor_path, output_path = (
        write_common_inputs(tmp_path)
    )

    count = biomedical_evidence_csv(
        context_path,
        evidence_path,
        output_path,
        model=FakeBiomedicalEncoder(),
        model_name="fake-biomedical-model",
        identity_path=identity_path,
        descriptor_path=descriptor_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert count == 1
    assert rows[0]["molecule_id"] == "mol_1"
    assert rows[0]["biomedical_model_name"] == "fake-biomedical-model"
    assert rows[0]["biomedical_model_status"] == "preferred_model_used"
    assert rows[0]["biomedical_evidence_status"] == "available"
    assert rows[0]["biomedical_similarity_score"] == "1.000"
    assert rows[0]["biomedical_relevance_category"] == "high_biomedical_relevance"
    assert rows[0]["biomedical_evidence_count"] == "2"
    assert rows[0]["top_biomedical_evidence_id"] == "ev_1"
    assert "Kinase signaling evidence" in rows[0]["top_biomedical_evidence_text"]
