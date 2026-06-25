import csv
from pathlib import Path

import pytest

import src.text_nlp as text_nlp_module
from src.text_nlp import (
    MODEL_NAME,
    MODEL_UNAVAILABLE_NOTE,
    ModelUnavailableError,
    OUTPUT_COLUMNS,
    RELEVANCE_QUERIES,
    categorize_relevance,
    main,
    score_rows,
    text_nlp_csv,
)


class MockSentenceModel:
    """Deterministic encoder for offline tests."""

    vectors = {
        "small molecule chemical structure public evidence reference context "
        "research prioritization": [1.0, 0.0, 0.0],
        "small molecule bioactivity evidence receptor modulation assay results "
        "drug discovery relevance": [0.0, 1.0, 0.0],
        "chemical novelty structural differentiation analog comparison generated "
        "molecule prioritization uniqueness": [0.0, 0.0, 1.0],
        "patent evidence": [0.9, 0.1, 0.0],
        "unrelated text": [0.1, 0.1, 0.1],
    }

    def encode(self, sentences, **kwargs):
        return [self.vectors[sentence] for sentence in sentences]


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.75, "high_nlp_relevance"),
        (0.749, "medium_nlp_relevance"),
        (0.55, "medium_nlp_relevance"),
        (0.549, "low_nlp_relevance"),
        (0.35, "low_nlp_relevance"),
        (0.349, "not_relevant"),
        (None, "not_relevant"),
    ],
)
def test_category_threshold_logic(
    score: float | None, expected: str
) -> None:
    assert categorize_relevance(score) == expected


def test_missing_text_handling() -> None:
    rows = [
        {
            "evidence_id": "missing",
            "molecule_id": "demo",
            "source_type": "synthetic",
            "title": "Missing text",
            "text": "",
        }
    ]

    result = score_rows(rows, MockSentenceModel())[0]

    assert result.evidence_id == "missing"
    assert result.max_relevance_score == ""
    assert result.nlp_relevance_category == "not_relevant"
    assert "missing" in result.nlp_notes.lower()


def write_input(path: Path) -> None:
    path.write_text(
        "evidence_id,molecule_id,source_type,title,text\n"
        "evidence_1,demo,synthetic,Patent example,patent evidence\n"
        "evidence_2,demo,synthetic,Missing example,\n",
        encoding="utf-8",
    )


def test_output_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "nlp.csv"
    write_input(input_path)

    text_nlp_csv(input_path, output_path, model=MockSentenceModel())

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS


def test_cli_writes_output_with_mock_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "nlp.csv"
    write_input(input_path)

    exit_code = main(
        ["--input", str(input_path), "--output", str(output_path)],
        model=MockSentenceModel(),
    )

    assert exit_code == 0
    assert output_path.exists()
    assert "Wrote 2 text NLP records" in capsys.readouterr().out
    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert rows[0]["model_name"] == MODEL_NAME
    assert rows[0]["nlp_relevance_category"] == "high_nlp_relevance"
    assert rows[1]["nlp_relevance_category"] == "not_relevant"


class FlexibleSentenceModel:
    """Deterministic encoder that accepts generated context strings."""

    def encode(self, sentences, **kwargs):
        vectors = {
            RELEVANCE_QUERIES[0]: [1.0, 0.0, 0.0],
            RELEVANCE_QUERIES[1]: [0.0, 1.0, 0.0],
            RELEVANCE_QUERIES[2]: [0.0, 0.0, 1.0],
        }
        return [vectors.get(sentence, [0.6, 0.7, 0.1]) for sentence in sentences]


def test_context_aware_nlp_writes_molecule_evidence_rows(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.csv"
    context_path = tmp_path / "compound_context.csv"
    molecules_path = tmp_path / "molecules.csv"
    descriptors_path = tmp_path / "descriptors.csv"
    output_path = tmp_path / "text_nlp.csv"
    evidence_path.write_text(
        "evidence_id,text,source,target_family,notes\n"
        "ev_1,Public kinase reference evidence.,local,protein_kinase,"
        "Grounded local note.\n",
        encoding="utf-8",
    )
    context_path.write_text(
        "molecule_id,exact_public_name,closest_public_compound,"
        "reported_targets,biological_reference_summary,"
        "biomedical_relevance_summary\n"
        "mol_1,,Public comparator,protein_kinase,"
        "Public reference summary.,Grounded biomedical context.\n",
        encoding="utf-8",
    )
    molecules_path.write_text(
        "molecule_id,smiles,compound_description,scaffold_family,notes\n"
        "mol_1,CCO,Generated local description,local scaffold,Local note\n",
        encoding="utf-8",
    )
    descriptors_path.write_text(
        "molecule_id,valid_smiles\nmol_1,True\n",
        encoding="utf-8",
    )

    count = text_nlp_csv(
        evidence_path,
        output_path,
        model=FlexibleSentenceModel(),
        context_path=context_path,
        molecule_path=molecules_path,
        descriptor_path=descriptors_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert count == 1
    assert rows[0]["molecule_id"] == "mol_1"
    assert rows[0]["evidence_id"] == "ev_1"
    assert rows[0]["nlp_status"] == "available"
    assert rows[0]["similarity_score"]
    assert "Public comparator" in rows[0]["molecule_text"]
    assert "Generated local description" in rows[0]["molecule_text"]
    assert "local scaffold" in rows[0]["molecule_text"]
    assert "mol_1" not in rows[0]["molecule_text"]


def test_model_unavailable_writes_valid_fallback_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence_path = tmp_path / "evidence.csv"
    context_path = tmp_path / "compound_context.csv"
    molecules_path = tmp_path / "molecules.csv"
    descriptors_path = tmp_path / "descriptors.csv"
    output_path = tmp_path / "text_nlp.csv"
    evidence_path.write_text(
        "evidence_id,text,source,target_family,notes\n"
        "ev_1,Public kinase reference evidence.,local,protein_kinase,"
        "Grounded local note.\n",
        encoding="utf-8",
    )
    context_path.write_text(
        "molecule_id,exact_public_name,closest_public_compound,"
        "reported_targets,biological_reference_summary\n"
        "mol_1,,Public comparator,protein_kinase,Public reference summary.\n",
        encoding="utf-8",
    )
    molecules_path.write_text(
        "molecule_id,smiles,compound_description\n"
        "mol_1,CCO,Generated local description\n",
        encoding="utf-8",
    )
    descriptors_path.write_text(
        "molecule_id,valid_smiles\nmol_1,True\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        text_nlp_module,
        "load_model",
        lambda model_name=MODEL_NAME: (_ for _ in ()).throw(
            ModelUnavailableError("missing model")
        ),
    )

    count = text_nlp_csv(
        evidence_path,
        output_path,
        context_path=context_path,
        molecule_path=molecules_path,
        descriptor_path=descriptors_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        rows = list(reader)
    assert count == 1
    assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS
    assert rows[0]["molecule_id"] == "mol_1"
    assert rows[0]["nlp_status"] == "model_unavailable"
    assert rows[0]["nlp_relevance_category"] == "not_run"
    assert rows[0]["max_relevance_score"] == "0.000"
    assert rows[0]["evidence_note"] == MODEL_UNAVAILABLE_NOTE
