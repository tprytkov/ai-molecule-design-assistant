import csv
from pathlib import Path

from src.scoring import scoring_csv
from src.standardize import standardize_csv
from src.structural import (
    add_structural_context_to_prioritization,
    docking_priority_label,
    normalize_docking_csv,
    structural_summary_csv,
)
from src.target import (
    DEMO_TARGET_PROFILE_PATH,
    TARGET_SPECIFIC_DEMO_DOCKING_PATH,
    TARGET_SPECIFIC_DEMO_MOLECULES_PATH,
    TARGET_SPECIFIC_DEMO_PROFILE_PATH,
    TARGET_SPECIFIC_DEMO_REFERENCES_PATH,
    load_target_profile,
    target_profile_csv,
)
from src.target.target_schema import (
    TARGET_SOURCE_GENERAL_DEMO,
    TARGET_SOURCE_TARGET_SPECIFIC_DEMO,
    TARGET_SOURCE_USER,
)


def write_structural_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "standardized": root / "standardized.csv",
        "descriptors": root / "descriptors.csv",
        "admet": root / "admet_summary.csv",
        "similarity": root / "similarity.csv",
        "public": root / "public_lookup.csv",
        "target": root / "target_profile_input.csv",
        "docking": root / "docking.csv",
    }
    paths["standardized"].write_text(
        "molecule_id,canonical_smiles,valid_smiles\n"
        "mol_a,CCO,True\n"
        "mol_b,CCN,True\n",
        encoding="utf-8",
    )
    paths["descriptors"].write_text(
        "molecule_id,canonical_smiles,valid_smiles,molecular_weight,logp,tpsa,"
        "hbd,hba,rotatable_bonds,qed,lipinski_violations,lipinski_pass,"
        "druglikeness_category\n"
        "mol_a,CCO,True,46.0,0.1,20.2,1,1,0,0.4,0,True,favorable\n"
        "mol_b,CCN,True,45.0,0.2,26.0,1,1,1,0.5,0,True,favorable\n",
        encoding="utf-8",
    )
    paths["admet"].write_text(
        "molecule_id,smiles,bbb_prediction_label,cns_property_flag,"
        "toxicity_risk_flag,admet_readiness_category,model_status,evidence_note\n"
        "mol_a,CCO,moderate,moderate,favorable,moderate,fallback_descriptor_rule,note\n",
        encoding="utf-8",
    )
    paths["similarity"].write_text(
        "molecule_id,canonical_smiles,valid_smiles,best_reference_id,"
        "best_reference_name,best_reference_smiles,tanimoto_similarity,"
        "similarity_category,similarity_error\n"
        "mol_a,CCO,True,ref_a,Reference A,CCO,0.420,structurally_distinct,\n"
        "mol_b,CCN,True,ref_b,Reference B,CCN,0.760,related_chemotype,\n",
        encoding="utf-8",
    )
    paths["public"].write_text(
        "molecule_id,source_database,match_type,public_id,similarity,lookup_status\n"
        "mol_a,PubChem,none,,,no_match\n",
        encoding="utf-8",
    )
    paths["target"].write_text(
        "target_id,target_name,gene_symbol,organism,uniprot_id,pdb_id,"
        "protein_structure_source,binding_site_description,disease_context,"
        "mechanism_context,reference_ligands,docking_protocol_note,"
        "target_relevance_note,disclaimer\n"
        "target_a,Demo kinase target,GENE,Homo sapiens,P00000,1ABC,"
        "user_provided,ATP pocket,Demo disease,Demo mechanism,lig_a,"
        "Vina demo protocol,User supplied target context,Research triage only.\n",
        encoding="utf-8",
    )
    paths["docking"].write_text(
        "molecule_id,smiles,target_id,docking_score,docking_rank,binding_site,"
        "pose_file,docking_program,docking_note\n"
        "mol_a,CCO,target_a,-9.4,1,ATP pocket,pose_a.pdbqt,Vina,Good pose\n"
        ",CCN,target_a,-7.4,2,ATP pocket,,Vina,Matched by SMILES\n"
        "mol_x,CCC,target_other,-8.2,3,ATP pocket,,Vina,Wrong target\n",
        encoding="utf-8",
    )
    return paths


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_target_profile_loading_and_output_generation(tmp_path: Path) -> None:
    paths = write_structural_inputs(tmp_path)
    output = tmp_path / "target_profile.csv"

    profile = target_profile_csv(output, source_path=paths["target"])

    assert profile.target_id == "target_a"
    assert profile.target_source == TARGET_SOURCE_USER
    rows = read_rows(output)
    assert rows[0]["target_name"] == "Demo kinase target"


def test_general_demo_target_profile_is_not_target_specific() -> None:
    profile = load_target_profile(DEMO_TARGET_PROFILE_PATH)

    assert profile.target_id == "demo_target_placeholder"
    assert profile.target_source == TARGET_SOURCE_GENERAL_DEMO
    assert "placeholder" in profile.target_name.lower()


def test_target_specific_demo_package_files_load_and_match(tmp_path: Path) -> None:
    profile = load_target_profile(TARGET_SPECIFIC_DEMO_PROFILE_PATH)
    standardized = tmp_path / "standardized.csv"
    docking = tmp_path / "docking_results_normalized.csv"

    standardize_csv(TARGET_SPECIFIC_DEMO_MOLECULES_PATH, standardized)
    count = normalize_docking_csv(
        TARGET_SPECIFIC_DEMO_DOCKING_PATH,
        docking,
        standardized_path=standardized,
        selected_target_id=profile.target_id,
    )

    references = read_rows(TARGET_SPECIFIC_DEMO_REFERENCES_PATH)
    docking_rows = read_rows(docking)
    assert profile.target_id == "adora2a_xanthine_demo"
    assert profile.target_source == TARGET_SOURCE_TARGET_SPECIFIC_DEMO
    assert profile.gene_symbol == "ADORA2A"
    assert profile.pdb_id == "3RFM"
    assert {row["reference_name"] for row in references} >= {"caffeine", "theophylline", "xanthine"}
    assert count == 3
    assert all(row["target_id"] == profile.target_id for row in docking_rows)
    assert all(row["target_docking_match"] == "True" for row in docking_rows)
    assert all(row["docking_status"] == "available" for row in docking_rows)


def test_general_demo_structural_output_is_not_target_available(tmp_path: Path) -> None:
    paths = write_structural_inputs(tmp_path)
    structural_summary_csv(
        target_output_path=tmp_path / "target_profile.csv",
        structural_properties_path=tmp_path / "structural_properties.csv",
        structural_prioritization_path=tmp_path / "structural_prioritization_inputs.csv",
        descriptors_path=paths["descriptors"],
        target_source_path=DEMO_TARGET_PROFILE_PATH,
        standardized_path=paths["standardized"],
    )

    rows = read_rows(tmp_path / "structural_prioritization_inputs.csv")
    assert rows[0]["target_available"] == "False"


def test_docking_normalization_matches_by_id_and_standardized_smiles(tmp_path: Path) -> None:
    paths = write_structural_inputs(tmp_path)
    output = tmp_path / "docking_results_normalized.csv"

    count = normalize_docking_csv(
        paths["docking"],
        output,
        standardized_path=paths["standardized"],
        selected_target_id="target_a",
    )

    rows = read_rows(output)
    assert count == 3
    assert rows[0]["molecule_id"] == "mol_a"
    assert rows[0]["target_docking_match"] == "True"
    assert rows[1]["molecule_id"] == "mol_b"
    assert rows[1]["docking_status"] == "available"
    assert rows[2]["docking_status"] == "target_mismatch"
    assert rows[2]["docking_available"] == "False"


def test_structural_outputs_and_prioritization_context_preserve_scores(tmp_path: Path) -> None:
    paths = write_structural_inputs(tmp_path)
    target_output = tmp_path / "target_profile.csv"
    docking_output = tmp_path / "docking_results_normalized.csv"
    structural = tmp_path / "structural_properties.csv"
    structural_inputs = tmp_path / "structural_prioritization_inputs.csv"

    counts = structural_summary_csv(
        target_output_path=target_output,
        structural_properties_path=structural,
        structural_prioritization_path=structural_inputs,
        descriptors_path=paths["descriptors"],
        admet_summary_path=paths["admet"],
        similarity_path=paths["similarity"],
        public_lookup_path=paths["public"],
        docking_input_path=paths["docking"],
        docking_output_path=docking_output,
        target_source_path=paths["target"],
        standardized_path=paths["standardized"],
    )

    assert counts == {"structural_properties": 2, "structural_prioritization_inputs": 2}
    structural_rows = read_rows(structural)
    assert structural_rows[0]["target_name"] == "Demo kinase target"
    assert structural_rows[0]["docking_priority_label"] == "strong_docking_signal"
    assert structural_rows[1]["docking_priority_label"] == "moderate_docking_signal"
    input_rows = read_rows(structural_inputs)
    assert input_rows[0]["target_available"] == "True"
    assert "computational triage only" in input_rows[0]["evidence_note"]

    prioritized = tmp_path / "prioritization_results.csv"
    scoring_csv(paths["descriptors"], paths["similarity"], prioritized)
    before = read_rows(prioritized)
    add_structural_context_to_prioritization(prioritized, structural)
    after = read_rows(prioritized)

    assert [row["prioritization_score"] for row in before] == [
        row["prioritization_score"] for row in after
    ]
    assert after[0]["docking_score"] == "-9.4"
    assert after[0]["target_id"] == "target_a"
    assert after[0]["docking_priority_label"] == "strong_docking_signal"
    assert "unchanged" in after[0]["structural_priority_note"]


def test_docking_priority_label_handles_missing_and_mismatch() -> None:
    assert docking_priority_label(None) == "docking_unavailable"
    assert docking_priority_label({"docking_status": "target_mismatch"}) == "target_mismatch"
    assert docking_priority_label({"docking_status": "available", "docking_score": "-6.0"}) == "weak_or_missing_docking_signal"
