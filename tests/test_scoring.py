import csv
import subprocess
import sys
from pathlib import Path

import pytest

from src.scoring import (
    NLP_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS,
    PUBLIC_OUTPUT_COLUMNS,
    PATENT_OUTPUT_COLUMNS,
    SURECHEMBL_OUTPUT_COLUMNS,
    PublicMatchSummary,
    PatentEvidenceSummary,
    SurechemblEvidenceSummary,
    aggregate_public_matches,
    aggregate_patent_evidence,
    aggregate_surechembl_evidence,
    aggregate_nlp_evidence,
    calculate_differentiation_score,
    calculate_lipinski_score,
    calculate_property_score,
    calculate_qed_score,
    categorize_prioritization,
    score_molecule,
    scoring_csv,
)


def valid_descriptor_row() -> dict[str, str]:
    return {
        "molecule_id": "demo",
        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "valid_smiles": "True",
        "molecular_weight": "180.159",
        "logp": "1.310",
        "tpsa": "63.600",
        "hbd": "1",
        "hba": "4",
        "rotatable_bonds": "2",
        "qed": "0.550",
        "lipinski_violations": "0",
        "lipinski_pass": "True",
        "descriptor_error": "",
    }


def valid_similarity_row() -> dict[str, str]:
    return {
        "molecule_id": "demo",
        "best_reference_name": "aspirin",
        "tanimoto_similarity": "0.400",
        "similarity_category": "structurally_distinct",
    }


def test_valid_high_quality_molecule_receives_nonzero_score() -> None:
    result = score_molecule(valid_descriptor_row(), valid_similarity_row())

    assert result.valid_smiles is True
    assert float(result.prioritization_score) > 0
    assert result.prioritization_category == "high_priority"


def test_invalid_molecule_is_retained_and_deprioritized() -> None:
    descriptor = valid_descriptor_row()
    descriptor.update(
        {
            "molecule_id": "invalid",
            "canonical_smiles": "",
            "valid_smiles": "False",
            "molecular_weight": "",
            "logp": "",
            "tpsa": "",
            "hbd": "",
            "hba": "",
            "rotatable_bonds": "",
            "qed": "",
            "lipinski_violations": "",
            "lipinski_pass": "",
            "descriptor_error": "Invalid generated structure.",
        }
    )

    result = score_molecule(descriptor, None)

    assert result.molecule_id == "invalid"
    assert result.valid_smiles is False
    assert result.prioritization_score == "0.000"
    assert result.prioritization_category == "deprioritized"
    assert "Invalid generated structure." in result.scoring_notes


def test_property_penalty_logic() -> None:
    perfect = calculate_property_score(
        True, 499.0, 5.0, 140.0, 5, 10, 10
    )
    two_violations = calculate_property_score(
        True, 501.0, 5.1, 140.0, 5, 10, 10
    )

    assert perfect == pytest.approx(1.0)
    assert two_violations == pytest.approx(4 / 6)
    assert calculate_property_score(
        False, 100.0, 1.0, 10.0, 0, 0, 0
    ) == 0.0


def test_qed_score_is_handled_correctly() -> None:
    assert calculate_qed_score(True, 0.72) == pytest.approx(0.72)
    assert calculate_qed_score(True, None) == 0.0
    assert calculate_qed_score(False, 0.72) == 0.0


@pytest.mark.parametrize(
    ("lipinski_pass", "violations", "expected"),
    [
        (True, 0, 1.0),
        (False, 1, 0.5),
        (False, 2, 0.25),
        (False, 3, 0.0),
        (False, None, 0.0),
    ],
)
def test_lipinski_score_categories(
    lipinski_pass: bool, violations: float | None, expected: float
) -> None:
    assert (
        calculate_lipinski_score(True, lipinski_pass, violations) == expected
    )
    assert calculate_lipinski_score(False, lipinski_pass, violations) == 0.0


def test_differentiation_rewards_lower_similarity() -> None:
    close_score = calculate_differentiation_score(True, 0.90)
    distinct_score = calculate_differentiation_score(True, 0.30)

    assert close_score == 0.25
    assert distinct_score == 1.0
    assert distinct_score > close_score
    assert calculate_differentiation_score(True, None) == 0.0


@pytest.mark.parametrize(
    ("score", "valid_smiles", "expected"),
    [
        (0.80, True, "high_priority"),
        (0.799, True, "medium_priority"),
        (0.60, True, "medium_priority"),
        (0.599, True, "low_priority"),
        (0.40, True, "low_priority"),
        (0.399, True, "deprioritized"),
        (0.95, False, "deprioritized"),
    ],
)
def test_final_prioritization_category_thresholds(
    score: float, valid_smiles: bool, expected: str
) -> None:
    assert categorize_prioritization(score, valid_smiles) == expected


def write_test_inputs(
    descriptor_path: Path, similarity_path: Path
) -> None:
    descriptor_path.write_text(
        "molecule_id,canonical_smiles,valid_smiles,molecular_weight,logp,"
        "tpsa,hbd,hba,rotatable_bonds,qed,lipinski_violations,"
        "lipinski_pass,descriptor_error\n"
        "demo,CC(=O)Oc1ccccc1C(=O)O,True,180.159,1.310,63.600,"
        "1,4,2,0.550,0,True,\n"
        "invalid,,False,,,,,,,,,,Invalid generated structure.\n",
        encoding="utf-8",
    )
    similarity_path.write_text(
        "molecule_id,best_reference_name,tanimoto_similarity,"
        "similarity_category\n"
        "demo,aspirin,1.000,very_close_analog\n"
        "invalid,,,not_available\n",
        encoding="utf-8",
    )


def test_output_columns_exist(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptors.csv"
    similarity_path = tmp_path / "similarity.csv"
    output_path = tmp_path / "prioritized.csv"
    write_test_inputs(descriptor_path, similarity_path)

    scoring_csv(descriptor_path, similarity_path, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS


def test_cli_writes_expected_output_file(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptors.csv"
    similarity_path = tmp_path / "similarity.csv"
    output_path = tmp_path / "prioritized.csv"
    write_test_inputs(descriptor_path, similarity_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.scoring",
            "--descriptors",
            str(descriptor_path),
            "--similarity",
            str(similarity_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert len(rows) == 2
    assert rows[0]["molecule_id"] == "demo"
    assert rows[1]["prioritization_category"] == "deprioritized"


def write_nlp_input(path: Path) -> None:
    path.write_text(
        "evidence_id,molecule_id,max_relevance_score,nlp_relevance_category\n"
        "evidence_1,demo,0.620,medium_nlp_relevance\n"
        "evidence_2,demo,0.910,high_nlp_relevance\n"
        "evidence_3,invalid,0.990,high_nlp_relevance\n",
        encoding="utf-8",
    )


def test_nlp_file_is_merged_using_maximum_score() -> None:
    summaries = aggregate_nlp_evidence(
        [
            {"molecule_id": "demo", "max_relevance_score": "0.620"},
            {"molecule_id": "demo", "max_relevance_score": "0.910"},
            {"molecule_id": "demo", "max_relevance_score": ""},
        ]
    )

    assert summaries["demo"].score == pytest.approx(0.91)
    assert summaries["demo"].category == "high_nlp_relevance"
    assert summaries["demo"].count == 3


def test_missing_nlp_evidence_is_not_run_and_score_is_neutral() -> None:
    result = score_molecule(
        valid_descriptor_row(), valid_similarity_row(), nlp_summary=None
    )

    assert result.nlp_status == "not_run"
    assert result.nlp_evidence_score == ""
    assert result.nlp_relevance_category == "not_run"
    assert result.nlp_evidence_count == "0"
    assert result.prioritization_score_with_nlp == result.prioritization_score


def test_invalid_molecule_remains_deprioritized_with_nlp() -> None:
    descriptor = valid_descriptor_row()
    descriptor["valid_smiles"] = "False"
    summary = aggregate_nlp_evidence(
        [{"molecule_id": "demo", "max_relevance_score": "0.990"}]
    )["demo"]

    result = score_molecule(
        descriptor, valid_similarity_row(), nlp_summary=summary
    )

    assert result.prioritization_category_with_nlp == "deprioritized"


def test_nlp_output_columns_exist(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptors.csv"
    similarity_path = tmp_path / "similarity.csv"
    nlp_path = tmp_path / "nlp.csv"
    output_path = tmp_path / "prioritized_with_nlp.csv"
    write_test_inputs(descriptor_path, similarity_path)
    write_nlp_input(nlp_path)

    scoring_csv(
        descriptor_path,
        similarity_path,
        output_path,
        nlp_path=nlp_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == NLP_OUTPUT_COLUMNS
        rows = list(reader)
    assert rows[0]["nlp_evidence_score"] == "0.910"
    assert rows[0]["nlp_status"] == "available"
    assert rows[0]["nlp_evidence_count"] == "2"
    assert rows[1]["prioritization_category_with_nlp"] == "deprioritized"


def test_empty_nlp_file_does_not_reduce_final_score(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptors.csv"
    similarity_path = tmp_path / "similarity.csv"
    nlp_path = tmp_path / "nlp.csv"
    output_path = tmp_path / "prioritized_with_nlp.csv"
    write_test_inputs(descriptor_path, similarity_path)
    nlp_path.write_text(
        "evidence_id,molecule_id,max_relevance_score,nlp_relevance_category\n",
        encoding="utf-8",
    )

    scoring_csv(
        descriptor_path,
        similarity_path,
        output_path,
        nlp_path=nlp_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert rows[0]["nlp_status"] == "not_run"
    assert rows[0]["nlp_evidence_score"] == ""
    assert rows[0]["nlp_relevance_category"] == "not_run"
    assert (
        rows[0]["prioritization_score_with_nlp"]
        == rows[0]["prioritization_score"]
    )


def test_cli_writes_output_with_nlp(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptors.csv"
    similarity_path = tmp_path / "similarity.csv"
    nlp_path = tmp_path / "nlp.csv"
    output_path = tmp_path / "prioritized_with_nlp.csv"
    write_test_inputs(descriptor_path, similarity_path)
    write_nlp_input(nlp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.scoring",
            "--descriptors",
            str(descriptor_path),
            "--similarity",
            str(similarity_path),
            "--nlp",
            str(nlp_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()


def write_public_lookup_input(path: Path) -> None:
    path.write_text(
        "molecule_id,source_database,match_type,public_id,similarity,"
        "lookup_status\n"
        "demo,PubChem,exact_inchikey,CID:2244,1.000,match_found\n"
        "demo,ChEMBL,similarity,CHEMBL1,0.990,match_found\n"
        "invalid,not_available,no_match,,,invalid_molecule\n",
        encoding="utf-8",
    )


def test_exact_public_match_sets_ip_flags() -> None:
    summary = PublicMatchSummary(
        known_public_match=True,
        exact_source="PubChem",
        exact_id="CID:2244",
        max_similarity=0.99,
    )
    result = score_molecule(
        valid_descriptor_row(),
        valid_similarity_row(),
        public_summary=summary,
    )

    assert result.known_public_match == "True"
    assert result.known_public_match_source == "PubChem"
    assert result.known_public_match_id == "CID:2244"
    assert result.novelty_flag == "known_public_compound"
    assert result.ip_potential_category == "low_ip_potential_signal"


def test_very_close_similarity_without_exact_match() -> None:
    summary = PublicMatchSummary(
        known_public_match=False,
        max_similarity=0.90,
    )
    result = score_molecule(
        valid_descriptor_row(),
        valid_similarity_row(),
        public_summary=summary,
    )

    assert result.known_public_match == "False"
    assert result.novelty_flag == "very_close_public_analog"
    assert result.ip_potential_category == "reduced_ip_potential_signal"


def test_missing_public_lookup_does_not_crash() -> None:
    result = score_molecule(
        valid_descriptor_row(),
        valid_similarity_row(),
        public_summary=None,
    )

    assert result.known_public_match == "False"
    assert result.novelty_flag == "not_available"
    assert result.ip_potential_category == "not_available"


def test_invalid_molecule_has_unavailable_ip_flags() -> None:
    descriptor = valid_descriptor_row()
    descriptor["valid_smiles"] = "False"
    result = score_molecule(
        descriptor,
        valid_similarity_row(),
        public_summary=PublicMatchSummary(True, "PubChem", "CID:1", 1.0),
    )

    assert result.novelty_flag == "not_available"
    assert result.ip_potential_category == "not_available"


def test_public_lookup_output_columns_and_aggregation(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptors.csv"
    similarity_path = tmp_path / "similarity.csv"
    nlp_path = tmp_path / "nlp.csv"
    public_path = tmp_path / "public.csv"
    output_path = tmp_path / "prioritized.csv"
    write_test_inputs(descriptor_path, similarity_path)
    write_nlp_input(nlp_path)
    write_public_lookup_input(public_path)

    scoring_csv(
        descriptor_path,
        similarity_path,
        output_path,
        nlp_path=nlp_path,
        public_lookup_path=public_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == PUBLIC_OUTPUT_COLUMNS
        rows = list(reader)
    assert rows[0]["known_public_match"] == "True"
    assert rows[0]["public_lookup_status"] == "match_found"
    assert rows[0]["pubchem_status"] == "match_found"
    assert rows[0]["chembl_status"] == "match_found"
    assert rows[0]["novelty_flag"] == "known_public_compound"
    assert rows[0]["ip_potential_category"] == "low_ip_potential_signal"

    summaries = aggregate_public_matches(
        [
            {
                "molecule_id": "demo",
                "source_database": "ChEMBL",
                "match_type": "similarity",
                "public_id": "CHEMBL1",
                "similarity": "0.900",
                "lookup_status": "match_found",
            }
        ]
    )
    assert summaries["demo"].max_similarity == pytest.approx(0.9)


def write_patent_input(path: Path) -> None:
    path.write_text(
        "molecule_id,search_status,patent_id,patent_title\n"
        "demo,match_found,123,Public text hit\n"
        "invalid,invalid_molecule,,\n",
        encoding="utf-8",
    )


def write_surechembl_input(path: Path) -> None:
    path.write_text(
        "molecule_id,tanimoto_similarity,lookup_status\n"
        "demo,1.000,match_found\n"
        "demo,0.321,match_found\n"
        "invalid,,invalid_molecule\n",
        encoding="utf-8",
    )


def test_patent_evidence_columns_and_aggregation(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptors.csv"
    similarity_path = tmp_path / "similarity.csv"
    nlp_path = tmp_path / "nlp.csv"
    public_path = tmp_path / "public.csv"
    patent_path = tmp_path / "patent.csv"
    output_path = tmp_path / "prioritized.csv"
    write_test_inputs(descriptor_path, similarity_path)
    write_nlp_input(nlp_path)
    write_public_lookup_input(public_path)
    write_patent_input(patent_path)

    scoring_csv(
        descriptor_path,
        similarity_path,
        output_path,
        nlp_path=nlp_path,
        public_lookup_path=public_path,
        patent_path=patent_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == PATENT_OUTPUT_COLUMNS
        rows = list(reader)
    assert rows[0]["patent_evidence_count"] == "1"
    assert rows[0]["patent_relevance_score"] == "1.000"
    assert rows[0]["patent_signal_category"] == "patent_text_evidence_signal"

    summaries = aggregate_patent_evidence(
        [{"molecule_id": "demo", "search_status": "offline"}]
    )
    assert summaries["demo"].category == "patent_search_not_available"


def test_score_molecule_accepts_patent_summary() -> None:
    result = score_molecule(
        valid_descriptor_row(),
        valid_similarity_row(),
        patent_summary=PatentEvidenceSummary(
            evidence_count=1,
            relevance_score=1.0,
            category="patent_text_evidence_signal",
            notes="Text evidence only.",
        ),
    )

    assert result.patent_evidence_count == "1"
    assert result.patent_signal_category == "patent_text_evidence_signal"


def test_surechembl_evidence_columns_and_aggregation(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptors.csv"
    similarity_path = tmp_path / "similarity.csv"
    nlp_path = tmp_path / "nlp.csv"
    public_path = tmp_path / "public.csv"
    patent_path = tmp_path / "patent.csv"
    surechembl_path = tmp_path / "surechembl.csv"
    output_path = tmp_path / "prioritized.csv"
    write_test_inputs(descriptor_path, similarity_path)
    write_nlp_input(nlp_path)
    write_public_lookup_input(public_path)
    write_patent_input(patent_path)
    write_surechembl_input(surechembl_path)

    scoring_csv(
        descriptor_path,
        similarity_path,
        output_path,
        nlp_path=nlp_path,
        public_lookup_path=public_path,
        patent_path=patent_path,
        surechembl_path=surechembl_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == SURECHEMBL_OUTPUT_COLUMNS
        rows = list(reader)
    assert rows[0]["surechembl_evidence_count"] == "2"
    assert rows[0]["surechembl_query_status"] == "match_found"
    assert rows[0]["chemberta_status"] == "not_run"
    assert rows[0]["surechembl_max_similarity"] == "1.000"
    assert (
        rows[0]["surechembl_signal_category"]
        == "very_close_patent_compound_signal"
    )

    summaries = aggregate_surechembl_evidence(
        [{"molecule_id": "demo", "tanimoto_similarity": "0.650", "lookup_status": "match_found"}]
    )
    assert summaries["demo"].category == "moderate_patent_chemistry_signal"


def test_score_molecule_accepts_surechembl_summary() -> None:
    result = score_molecule(
        valid_descriptor_row(),
        valid_similarity_row(),
        surechembl_summary=SurechemblEvidenceSummary(
            evidence_count=1,
            max_similarity=0.92,
            category="very_close_patent_compound_signal",
            notes="Patent-associated compound similarity signal.",
        ),
    )

    assert result.surechembl_evidence_count == "1"
    assert result.surechembl_max_similarity == "0.920"
    assert (
        result.surechembl_signal_category
        == "very_close_patent_compound_signal"
    )
