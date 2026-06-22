from pathlib import Path

from src.compound_context import OUTPUT_COLUMNS, build_context_rows, write_context_csv
from src.compound_qa import CompoundEvidence, full_report
from src.text_nlp import RELEVANCE_QUERIES, score_rows


MOLECULE = {
    "molecule_id": "mol_1",
    "canonical_smiles": "CCO",
}


def test_exact_match_populates_name_and_public_ids(tmp_path: Path) -> None:
    public_rows = [
        {
            "molecule_id": "mol_1",
            "source_database": "PubChem",
            "match_type": "exact_inchikey",
            "public_id": "CID:702",
            "public_name": "Ethanol",
            "iupac_name": "ethanol",
            "common_names": "ethyl alcohol",
            "similarity": "1.000",
            "lookup_status": "match_found",
        },
        {
            "molecule_id": "mol_1",
            "source_database": "ChEMBL",
            "match_type": "similarity",
            "public_id": "CHEMBL545",
            "public_name": "Ethanol",
            "similarity": "1.000",
            "lookup_status": "match_found",
        },
    ]

    result = build_context_rows([MOLECULE], public_rows, [])[0]

    assert result.identity_status == "exact_public_match"
    assert result.exact_public_name == "Ethanol"
    assert result.iupac_name == "ethanol"
    assert result.common_names == "ethyl alcohol"
    assert result.pubchem_cid == "702"
    assert result.chembl_id == "CHEMBL545"
    assert result.context_status == "exact_public_context"

    output = tmp_path / "compound_context.csv"
    write_context_csv(output, [result])
    assert output.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        OUTPUT_COLUMNS
    )


def build_reference_context(similarity: str):
    similarity_rows = [
        {
            "molecule_id": "mol_1",
            "reference_id": "ref_1",
            "reference_name": "Public comparator",
            "reference_source": "public_reference_panel",
            "tanimoto_similarity": similarity,
        }
    ]
    reference_rows = [
        {
            "reference_id": "ref_1",
            "reference_name": "Public comparator",
            "target_family": "protein_kinase",
            "notes": "Public reference annotation.",
        }
    ]

    return build_context_rows(
        [MOLECULE], [], similarity_rows, reference_rows
    )[0]


def test_similarity_055_gives_similar_reference_context() -> None:
    result = build_reference_context("0.550")

    assert result.exact_public_name == ""
    assert result.closest_public_compound == "Public comparator"
    assert result.closest_public_similarity == "0.550"
    assert "protein_kinase" in result.reported_targets
    assert "not established for the query molecule" in result.reported_targets
    assert result.context_status == "similar_reference_context"
    assert result.context_confidence == "moderate"


def test_similarity_035_gives_weak_similar_reference_context() -> None:
    result = build_reference_context("0.350")

    assert result.context_status == "weak_similar_reference_context"
    assert result.context_confidence == "low"
    assert "Weak reference-only context" in result.reported_targets
    assert "protein_kinase" in result.reported_targets
    assert "weak, reference-only context" in result.biomedical_relevance_summary


def test_similarity_018_gives_structural_context_only_without_targets() -> None:
    result = build_reference_context("0.180")

    assert result.closest_public_compound == "Public comparator"
    assert result.closest_public_similarity == "0.180"
    assert result.context_status == "structural_context_only"
    assert result.context_confidence == "very_low"
    assert result.reported_targets == ""
    assert result.reported_assays == ""
    assert result.biological_reference_summary == ""
    assert result.biomedical_relevance_summary == (
        "No reliable biomedical context was assigned because the closest "
        "reference similarity was below the reporting threshold."
    )


def test_no_evidence_gives_no_public_context_without_invented_name() -> None:
    result = build_context_rows([MOLECULE], [], [])[0]

    assert result.context_status == "no_public_context"
    assert result.exact_public_name == ""
    assert result.closest_public_compound == ""
    assert result.reported_targets == ""
    assert result.reported_assays == ""
    assert (
        result.biomedical_relevance_summary
        == "No public biomedical context was found in the available local "
        "lookup or reference evidence."
    )


def test_report_includes_public_biomedical_context_section() -> None:
    evidence = CompoundEvidence(
        prioritized={
            "molecule_id": "mol_1",
            "canonical_smiles": "CCO",
            "valid_smiles": "True",
        },
        descriptor={},
        similarity_hits=(),
        public_lookup=(),
        compound_context={
            "identity_status": "similar_reference",
            "closest_public_compound": "Public comparator",
            "closest_public_similarity": "0.720",
            "biomedical_relevance_summary": (
                "No exact public identity was found. The closest grounded "
                "reference is Public comparator."
            ),
            "context_status": "similar_reference_context",
        },
        nlp_evidence=(),
        surechembl_evidence=(),
        visualization_row={},
        visualization_rows=(),
    )

    report = full_report(evidence)

    assert "## Public biomedical context" in report
    assert "Public comparator" in report
    assert "similar_reference_context" in report


def test_report_includes_low_similarity_orientation_warning() -> None:
    evidence = CompoundEvidence(
        prioritized={
            "molecule_id": "mol_1",
            "canonical_smiles": "CCO",
            "valid_smiles": "True",
        },
        descriptor={},
        similarity_hits=(),
        public_lookup=(),
        compound_context={
            "identity_status": "similar_reference",
            "closest_public_compound": "Public comparator",
            "closest_public_similarity": "0.180",
            "reported_targets": "",
            "context_confidence": "very_low",
            "context_status": "structural_context_only",
            "biomedical_relevance_summary": (
                "No reliable biomedical context was assigned because the "
                "closest reference similarity was below the reporting threshold."
            ),
        },
        nlp_evidence=(),
        surechembl_evidence=(),
        visualization_row={},
        visualization_rows=(),
    )

    report = full_report(evidence)

    assert "Biological target context assigned: No" in report
    assert (
        "The nearest public/reference compound is shown for orientation only. "
        "Similarity is too low to transfer biological target context."
    ) in report


class RecordingSentenceModel:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, sentences, **kwargs):
        self.encoded.extend(sentences)
        vectors = {
            RELEVANCE_QUERIES[0]: [1.0, 0.0, 0.0],
            RELEVANCE_QUERIES[1]: [0.0, 1.0, 0.0],
            RELEVANCE_QUERIES[2]: [0.0, 0.0, 1.0],
        }
        return [vectors.get(sentence, [0.5, 0.5, 0.5]) for sentence in sentences]


def test_text_nlp_uses_context_text_not_molecule_id_alone() -> None:
    model = RecordingSentenceModel()
    rows = [
        {
            "evidence_id": "e1",
            "molecule_id": "opaque_internal_id",
            "source_type": "local",
            "title": "Grounded evidence",
            "text": "Observed public evidence.",
        }
    ]
    context = {
        "opaque_internal_id": {
            "molecule_id": "opaque_internal_id",
            "exact_public_name": "Grounded public name",
            "reported_targets": "reported target context",
            "biomedical_relevance_summary": "Grounded biomedical summary.",
        }
    }

    score_rows(rows, model, context_by_molecule=context)

    scored_text = model.encoded[-1]
    assert "Grounded public name" in scored_text
    assert "reported target context" in scored_text
    assert "Grounded biomedical summary." in scored_text
    assert scored_text != "opaque_internal_id"
