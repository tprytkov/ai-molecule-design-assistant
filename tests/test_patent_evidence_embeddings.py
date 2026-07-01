import csv
from pathlib import Path

import pytest

import src.patent_evidence_embeddings as patent_module
from src.patent_evidence_embeddings import (
    OUTPUT_COLUMNS,
    PATENT_MODEL_UNAVAILABLE_NOTE,
    PatentModelUnavailableError,
    patent_evidence_embeddings_csv,
)


class FakePatentEncoder:
    """Deterministic encoder for patent/IP-context tests."""

    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            text = str(sentence).lower()
            if "patent kinase" in text or "surechembl" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "unrelated" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.8, 0.2, 0.0])
        return vectors


def write_common_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    surechembl_path = tmp_path / "surechembl_evidence.csv"
    public_lookup_path = tmp_path / "public_lookup.csv"
    identity_path = tmp_path / "chemical_identity.csv"
    context_path = tmp_path / "compound_context.csv"
    patent_text_path = tmp_path / "patent_text.csv"
    output_path = tmp_path / "patent_evidence_embeddings.csv"
    surechembl_path.write_text(
        "molecule_id,lookup_status,surechembl_id,compound_name,patent_id,"
        "patent_number,patent_title,patent_date,source_section,"
        "patent_metadata_status,evidence_note\n"
        "mol_1,match_found,SCHEMBL1,Patent kinase compound,US-1,US1,"
        "SureChEMBL patent kinase document,2024-01-01,claims,available,"
        "SureChEMBL structure evidence.\n",
        encoding="utf-8",
    )
    public_lookup_path.write_text(
        "molecule_id,source_database,public_name,lookup_status\n"
        "mol_1,PubChem,Demo kinase molecule,match_found\n",
        encoding="utf-8",
    )
    identity_path.write_text(
        "molecule_id,exact_public_name,preferred_name\nmol_1,,Demo kinase\n",
        encoding="utf-8",
    )
    context_path.write_text(
        "molecule_id,closest_public_compound,biomedical_relevance_summary\n"
        "mol_1,Patent kinase comparator,Patent kinase context.\n",
        encoding="utf-8",
    )
    patent_text_path.write_text(
        "evidence_id,molecule_id,text\n"
        "pat_ev_1,mol_1,Patent kinase composition evidence.\n"
        "pat_ev_2,mol_1,Unrelated document text.\n",
        encoding="utf-8",
    )
    return (
        surechembl_path,
        public_lookup_path,
        identity_path,
        context_path,
        patent_text_path,
        output_path,
    )


def test_patent_evidence_fallback_writes_valid_csv_when_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        surechembl_path,
        public_lookup_path,
        identity_path,
        context_path,
        patent_text_path,
        output_path,
    ) = write_common_inputs(tmp_path)
    monkeypatch.setattr(
        patent_module,
        "load_model",
        lambda model_name=patent_module.DEFAULT_PATENT_MODEL: (_ for _ in ()).throw(
            PatentModelUnavailableError("missing")
        ),
    )

    count = patent_evidence_embeddings_csv(
        surechembl_path,
        output_path,
        public_lookup_path=public_lookup_path,
        identity_path=identity_path,
        context_path=context_path,
        patent_text_path=patent_text_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        rows = list(reader)
    assert count == 1
    assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS
    assert rows[0]["molecule_id"] == "mol_1"
    assert rows[0]["patent_model_status"] == "model_unavailable"
    assert rows[0]["patent_evidence_status"] == "skipped"
    assert rows[0]["patent_similarity_score"] == "0.000"
    assert rows[0]["patent_relevance_category"] == "not_run"
    assert rows[0]["surechembl_structure_status"] == "match_found"
    assert rows[0]["patent_document_metadata_status"] == "available"
    assert rows[0]["evidence_note"] == PATENT_MODEL_UNAVAILABLE_NOTE


def test_patent_evidence_scores_with_fake_encoder(tmp_path: Path) -> None:
    (
        surechembl_path,
        public_lookup_path,
        identity_path,
        context_path,
        patent_text_path,
        output_path,
    ) = write_common_inputs(tmp_path)

    count = patent_evidence_embeddings_csv(
        surechembl_path,
        output_path,
        public_lookup_path=public_lookup_path,
        identity_path=identity_path,
        context_path=context_path,
        patent_text_path=patent_text_path,
        model=FakePatentEncoder(),
        model_name="fake-paecter-model",
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert count == 1
    assert rows[0]["patent_model_name"] == "fake-paecter-model"
    assert rows[0]["patent_model_status"] == "preferred_model_used"
    assert rows[0]["patent_evidence_status"] == "available"
    assert rows[0]["patent_similarity_score"] == "1.000"
    assert rows[0]["patent_relevance_category"] == "high_patent_relevance"
    assert rows[0]["patent_evidence_count"] == "3"
    assert rows[0]["top_patent_evidence_id"] == "US-1"
    assert "patent kinase" in rows[0]["top_patent_evidence_text"].lower()
