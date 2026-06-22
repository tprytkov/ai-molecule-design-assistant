import csv
from pathlib import Path

from src.chemical_identity import (
    ChemicalIdentity,
    identify_rows,
    write_identity_csv,
)
from src.compound_qa import CompoundEvidence, full_report
from src.public_lookup import PublicLookupHttpError


class MockIdentityClient:
    def __init__(self, *, pubchem_404=False, cactus_name="") -> None:
        self.pubchem_404 = pubchem_404
        self.cactus_name = cactus_name

    def get_json(self, url: str, timeout: float):
        if "/synonyms/" in url:
            return {
                "InformationList": {
                    "Information": [
                        {"CID": 702, "Synonym": ["Ethanol", "ethyl alcohol"]}
                    ]
                }
            }
        if self.pubchem_404:
            raise PublicLookupHttpError(404, "not found")
        return {
            "PropertyTable": {
                "Properties": [
                    {
                        "CID": 702,
                        "Title": "Ethanol",
                        "IUPACName": "ethanol",
                    }
                ]
            }
        }

    def get_text(self, url: str, timeout: float) -> str:
        return self.cactus_name


def standardized(smiles="CCO", valid="True"):
    return [
        {
            "molecule_id": "demo",
            "smiles": smiles,
            "canonical_smiles": smiles if valid == "True" else "",
            "valid_smiles": valid,
            "error_message": "" if valid == "True" else "invalid",
        }
    ]


def test_valid_smiles_gets_inchikey() -> None:
    result = identify_rows(standardized())[0]

    assert result.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert result.valid_smiles is True


def test_invalid_smiles_handled() -> None:
    result = identify_rows(standardized("C1CC", "False"))[0]

    assert result.identity_status == "invalid_smiles"
    assert result.lookup_status == "invalid_smiles"
    assert result.inchikey == ""


def test_mocked_pubchem_exact_match_populates_name_and_cid() -> None:
    result = identify_rows(
        standardized(), online=True, client=MockIdentityClient()
    )[0]

    assert result.identity_status == "exact_public_identity"
    assert result.pubchem_cid == "702"
    assert result.exact_public_name == "Ethanol"
    assert result.iupac_name == "ethanol"
    assert "ethyl alcohol" in result.synonyms
    assert result.name_source == "PubChem"
    assert result.lookup_status == "match_found"


def test_mocked_pubchem_404_is_no_public_identity_no_match() -> None:
    result = identify_rows(
        standardized(),
        online=True,
        client=MockIdentityClient(pubchem_404=True),
        use_cactus=False,
    )[0]

    assert result.identity_status == "no_public_identity"
    assert result.lookup_status == "no_match"
    assert result.error_message == ""


def test_mocked_cactus_fallback_is_generated_iupac_name_only() -> None:
    result = identify_rows(
        standardized(),
        online=True,
        client=MockIdentityClient(
            pubchem_404=True, cactus_name="ethanol"
        ),
    )[0]

    assert result.identity_status == "generated_iupac_name_only"
    assert result.iupac_name == "ethanol"
    assert result.exact_public_name == ""
    assert result.preferred_name == ""
    assert result.name_source == "NCI Cactus"
    assert result.lookup_status == "no_match"


def test_no_invented_names_when_no_source_returns_one() -> None:
    result = identify_rows(
        standardized(),
        online=True,
        client=MockIdentityClient(pubchem_404=True),
    )[0]

    assert result.exact_public_name == ""
    assert result.preferred_name == ""
    assert result.iupac_name == ""
    assert result.synonyms == ""


def test_identity_csv_column_order(tmp_path: Path) -> None:
    output = tmp_path / "chemical_identity.csv"
    write_identity_csv(
        output,
        [
            ChemicalIdentity(
                molecule_id="demo",
                smiles="CCO",
                valid_smiles=True,
                inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            )
        ],
    )

    with output.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        row = next(reader)
    assert list(row) == [
        "molecule_id",
        "smiles",
        "valid_smiles",
        "inchikey",
        "identity_status",
        "pubchem_cid",
        "chembl_id",
        "exact_public_name",
        "iupac_name",
        "preferred_name",
        "synonyms",
        "name_source",
        "identity_confidence",
        "lookup_status",
        "error_message",
    ]


def test_report_includes_chemical_identity() -> None:
    evidence = CompoundEvidence(
        prioritized={
            "molecule_id": "demo",
            "valid_smiles": "True",
            "canonical_smiles": "CCO",
        },
        descriptor={},
        similarity_hits=(),
        public_lookup=(),
        compound_context={},
        nlp_evidence=(),
        surechembl_evidence=(),
        visualization_row={},
        visualization_rows=(),
        chemical_identity={
            "identity_status": "exact_public_identity",
            "exact_public_name": "Ethanol",
            "iupac_name": "ethanol",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "pubchem_cid": "702",
            "name_source": "PubChem",
            "identity_confidence": "high",
            "lookup_status": "match_found",
        },
    )

    report = full_report(evidence)

    assert "## Chemical identity" in report
    assert "Exact public name: Ethanol" in report


def test_app_imports_successfully() -> None:
    import app

    assert hasattr(app, "run_app")
