from pathlib import Path

import pytest

from src.compound_qa import (
    closest_compounds_answer,
    compound_qa,
    ip_potential_answer,
    is_known_answer,
    load_compound_evidence,
    read_csv,
)


def write_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "prioritized": root / "prioritized.csv",
        "similarity": root / "similarity.csv",
        "public": root / "public.csv",
        "nlp": root / "nlp.csv",
        "descriptors": root / "descriptors.csv",
    }
    paths["prioritized"].write_text(
        "molecule_id,canonical_smiles,valid_smiles,validity_score,"
        "property_score,qed_score,lipinski_score,differentiation_score,"
        "prioritization_score,prioritization_category,nlp_evidence_score,"
        "nlp_relevance_category,nlp_evidence_count,"
        "prioritization_score_with_nlp,prioritization_category_with_nlp,"
        "known_public_match,known_public_match_source,known_public_match_id,"
        "novelty_flag,ip_potential_category,ip_potential_notes,"
        "public_lookup_status,pubchem_status,chembl_status,nlp_status,"
        "surechembl_query_status,chemberta_status,"
        "chemberta_model,chemberta_embedding_available,chemberta_embedding_dim,"
        "patent_evidence_count,patent_max_similarity,patent_relevance_score,"
        "patent_signal_category,patent_signal_notes,"
        "surechembl_evidence_count,surechembl_max_similarity,"
        "surechembl_signal_category,surechembl_signal_notes\n"
        "demo,CCO,True,1.000,1.000,0.500,1.000,0.250,0.750,"
        "medium_priority,0.800,high_nlp_relevance,1,0.760,medium_priority,"
        "True,PubChem,CID:702,known_public_compound,"
        "low_ip_potential_signal,Exact public match reduces the research signal.,"
        "match_found,match_found,match_found,available,match_found,available,"
        "mock-chemberta,True,3,"
        "1,1.000,1.000,patent_text_evidence_signal,Patent text match found.,"
        "1,1.000,very_close_patent_compound_signal,"
        "Very close patent-associated compound signal.\n"
        "invalid,,False,0,0,0,0,0,0,deprioritized,0,not_relevant,0,0,"
        "deprioritized,False,,,not_available,not_available,"
        "Signal unavailable for invalid molecule.,0,,0.000,not_available,"
        "not_available,not_available,not_available,not_run,invalid_molecule,not_run,"
        ",False,,"
        "Patent evidence unavailable.,0,,not_available,"
        "SureChEMBL-style evidence unavailable.\n",
        encoding="utf-8",
    )
    paths["descriptors"].write_text(
        "molecule_id,canonical_smiles,valid_smiles,molecular_weight,logp,"
        "tpsa,hbd,hba,rotatable_bonds,qed,lipinski_pass,descriptor_error\n"
        "demo,CCO,True,46.069,-0.001,20.230,1,1,0,0.407,True,\n"
        "invalid,,False,,,,,,,,,Invalid structure.\n",
        encoding="utf-8",
    )
    paths["similarity"].write_text(
        "molecule_id,hit_rank,reference_name,tanimoto_similarity,"
        "similarity_category,evidence_note\n"
        "demo,1,ethanol,1.000,very_close_analog,Public-safe reference.\n",
        encoding="utf-8",
    )
    paths["public"].write_text(
        "molecule_id,source_database,match_type,public_id,public_name,"
        "similarity,lookup_status,error_message\n"
        "demo,PubChem,exact_inchikey,CID:702,Ethanol,1.000,match_found,\n"
        "demo,ChEMBL,similarity,CHEMBL123,Ethanol,0.990,match_found,\n",
        encoding="utf-8",
    )
    paths["nlp"].write_text(
        "molecule_id,title,max_relevance_score,nlp_relevance_category\n"
        "demo,Public evidence,0.800,high_nlp_relevance\n",
        encoding="utf-8",
    )
    paths["surechembl"] = root / "surechembl.csv"
    paths["surechembl"].write_text(
        "molecule_id,canonical_smiles,valid_smiles,surechembl_id,patent_id,"
        "patent_title,patent_date,compound_name,patent_compound_smiles,"
        "tanimoto_similarity,similarity_category,source_section,"
        "evidence_note,lookup_status,error_message\n"
        "demo,CCO,True,SC-ETH,DEMO-PATENT-ETHANOL-001,"
        "Synthetic demonstration record,2026-01-01,ethanol,CCO,1.000,"
        "very_close_patent_analog,claims_demo,"
        "Public-safe demonstration patent-associated compound evidence.,"
        "match_found,\n",
        encoding="utf-8",
    )
    paths["visualization"] = root / "visualization_coordinates.csv"
    paths["visualization"].write_text(
        "molecule_id,x,y,coordinate_method,prioritization_score_with_nlp,"
        "novelty_flag,ip_potential_category,known_public_match,"
        "best_reference_name,tanimoto_similarity,cluster_id\n"
        "demo,0.100,0.200,umap,0.760,known_public_compound,"
        "low_ip_potential_signal,True,ethanol,1.000,0\n"
        "neighbor,0.200,0.300,umap,0.700,not_available,"
        "not_available,False,ethanol,0.500,0\n",
        encoding="utf-8",
    )
    return paths


def load(root: Path, molecule_id: str = "demo"):
    paths = write_inputs(root)
    return load_compound_evidence(
        molecule_id,
        prioritized_path=paths["prioritized"],
        similarity_path=paths["similarity"],
        public_lookup_path=paths["public"],
        nlp_path=paths["nlp"],
        descriptor_path=paths["descriptors"],
        surechembl_path=paths["surechembl"],
    )


def test_molecule_found(tmp_path: Path) -> None:
    evidence = load(tmp_path)
    assert evidence.prioritized["molecule_id"] == "demo"


def test_missing_molecule_gives_clear_error(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    with pytest.raises(ValueError, match="Molecule ID 'missing' was not found"):
        load_compound_evidence(
            "missing",
            prioritized_path=paths["prioritized"],
            similarity_path=paths["similarity"],
            public_lookup_path=paths["public"],
            nlp_path=paths["nlp"],
            descriptor_path=paths["descriptors"],
            surechembl_path=paths["surechembl"],
        )


def test_full_report_creates_markdown_file(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    output = tmp_path / "report.md"
    answer = compound_qa(
        "demo",
        "full_report",
        output,
        prioritized_path=paths["prioritized"],
        similarity_path=paths["similarity"],
        public_lookup_path=paths["public"],
        nlp_path=paths["nlp"],
        descriptor_path=paths["descriptors"],
        surechembl_path=paths["surechembl"],
        visualization_path=paths["visualization"],
        image_dir=tmp_path / "report_images",
    )
    assert output.exists()
    assert "# Compound Intelligence Report: demo" in answer
    assert "Why It Was Ranked" in answer
    assert "Follow-up Review Interpretation" in answer
    assert "![2D structure for demo](report_images/demo.png)" in answer
    assert (tmp_path / "report_images" / "demo.png").exists()
    assert "Evidence completeness" in answer
    assert "| PubChem status | match_found | queried; evidence found |" in answer
    assert "Patent Evidence Summary" not in answer
    assert "patent_text_evidence_signal" not in answer
    assert "SureChEMBL Structure Evidence" in answer
    assert "very_close_patent_compound_signal" in answer
    assert "PatentsView" not in answer
    assert "USPTO" not in answer
    for banned in ("patentability", "IP-potential", "patentable", "not patentable"):
        assert banned.lower() not in answer.lower()


def test_is_known_uses_pubchem_and_chembl_results(tmp_path: Path) -> None:
    answer = is_known_answer(load(tmp_path))
    assert "PubChem CID:702" in answer
    assert "ChEMBL CHEMBL123" in answer


def test_closest_compounds_uses_both_sources(tmp_path: Path) -> None:
    answer = closest_compounds_answer(load(tmp_path))
    assert "Public Database Results" in answer
    assert "CHEMBL123" in answer
    assert "Local Reference Similarity Hits" in answer
    assert "ethanol" in answer


def test_invalid_molecule_is_handled_clearly(tmp_path: Path) -> None:
    evidence = load(tmp_path, molecule_id="invalid")
    assert "invalid" in is_known_answer(evidence).lower()
    assert "invalid" in ip_potential_answer(evidence).lower()


def test_report_includes_known_match_and_low_ip_signal(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    output = tmp_path / "report.md"
    answer = compound_qa(
        "demo",
        "full_report",
        output,
        prioritized_path=paths["prioritized"],
        similarity_path=paths["similarity"],
        public_lookup_path=paths["public"],
        nlp_path=paths["nlp"],
        descriptor_path=paths["descriptors"],
        surechembl_path=paths["surechembl"],
        visualization_path=paths["visualization"],
    )

    assert "Known public match: True" in answer
    assert "known_public_compound" in answer
    assert "reduces the chemical differentiation" in answer


def test_report_includes_surechembl_evidence_section(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    output = tmp_path / "report.md"
    answer = compound_qa(
        "demo",
        "full_report",
        output,
        prioritized_path=paths["prioritized"],
        similarity_path=paths["similarity"],
        public_lookup_path=paths["public"],
        nlp_path=paths["nlp"],
        descriptor_path=paths["descriptors"],
        surechembl_path=paths["surechembl"],
        visualization_path=paths["visualization"],
    )

    assert "SureChEMBL Structure Evidence" in answer
    assert "DEMO-PATENT-ETHANOL-001" in answer
    assert "Public-safe demonstration patent-associated compound evidence" in answer


def test_report_shows_text_evidence_not_run(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    rows = paths["prioritized"].read_text(encoding="utf-8").splitlines()
    rows[1] = rows[1].replace(",available,", ",not_run,")
    paths["prioritized"].write_text("\n".join(rows) + "\n", encoding="utf-8")
    paths["nlp"].write_text(
        "molecule_id,title,max_relevance_score,nlp_relevance_category\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.md"

    answer = compound_qa(
        "demo",
        "full_report",
        output,
        prioritized_path=paths["prioritized"],
        similarity_path=paths["similarity"],
        public_lookup_path=paths["public"],
        nlp_path=paths["nlp"],
        descriptor_path=paths["descriptors"],
        surechembl_path=paths["surechembl"],
        visualization_path=paths["visualization"],
    )

    assert "Text-evidence scoring was not run for this workflow." in answer
    assert "not_relevant" not in answer


def test_report_includes_weak_reference_support_sentence(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    paths["similarity"].write_text(
        "molecule_id,hit_rank,reference_name,tanimoto_similarity,"
        "similarity_category,evidence_note\n"
        "demo,1,ethanol,0.210,structurally_distinct,Public-safe reference.\n",
        encoding="utf-8",
    )

    answer = compound_qa(
        "demo",
        "full_report",
        tmp_path / "report.md",
        prioritized_path=paths["prioritized"],
        similarity_path=paths["similarity"],
        public_lookup_path=paths["public"],
        nlp_path=paths["nlp"],
        descriptor_path=paths["descriptors"],
        surechembl_path=paths["surechembl"],
        visualization_path=paths["visualization"],
        image_dir=tmp_path / "report_images",
    )

    assert (
        "This molecule is property-favorable and chemically differentiated, "
        "but has weak RDKit fingerprint similarity to the uploaded reference panel."
    ) in answer


def test_report_includes_chemberta_umap_context_and_cluster_note(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path)

    answer = compound_qa(
        "demo",
        "full_report",
        tmp_path / "report.md",
        prioritized_path=paths["prioritized"],
        similarity_path=paths["similarity"],
        public_lookup_path=paths["public"],
        nlp_path=paths["nlp"],
        descriptor_path=paths["descriptors"],
        surechembl_path=paths["surechembl"],
        visualization_path=paths["visualization"],
        image_dir=tmp_path / "report_images",
    )

    assert "ChemBERTa/UMAP coordinates are available" in answer
    assert "Coordinate method: umap" in answer
    assert "Research-prioritization score on the chemical-space map" in answer
    assert (
        "Clustering was not informative for this run because all molecules were assigned to one cluster."
        in answer
    )
    assert "Nearest generated neighbors on the 2D map" in answer


def test_legacy_filename_fallback_reads_old_outputs(tmp_path: Path) -> None:
    legacy = tmp_path / "prioritized_with_nlp_demo.csv"
    legacy.write_text(
        "molecule_id,valid_smiles\n"
        "legacy,True\n",
        encoding="utf-8",
    )

    rows = read_csv(tmp_path / "prioritization_results.csv")

    assert rows[0]["molecule_id"] == "legacy"
