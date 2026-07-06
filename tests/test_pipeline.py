import csv
from pathlib import Path

import pytest

from src.pipeline import PipelinePaths, main, run_pipeline
from src.surechembl_lookup import SURECHEMBL_DOCUMENTS_FOR_STRUCTURES_URL
from src.text_nlp import RELEVANCE_QUERIES


class MockSentenceModel:
    """Deterministic local encoder for end-to-end pipeline tests."""

    def encode(self, sentences, **kwargs):
        vectors = {
            RELEVANCE_QUERIES[0]: [1.0, 0.0, 0.0],
            RELEVANCE_QUERIES[1]: [0.0, 1.0, 0.0],
            RELEVANCE_QUERIES[2]: [0.0, 0.0, 1.0],
        }
        return [vectors.get(sentence, [0.8, 0.2, 0.1]) for sentence in sentences]


class MockChembertaEmbedder:
    """Deterministic ChemBERTa-like embedder for pipeline tests."""

    model_name = "mock-chemberta"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, smiles: str) -> list[float]:
        self.calls += 1
        return [float(len(smiles)), float(smiles.count("C")), 1.0]


class MockLookupClient:
    """Track and mock public lookup calls."""

    def __init__(self) -> None:
        self.calls = 0

    def get_json(self, url: str, timeout: float):
        self.calls += 1
        if "pubchem" in url:
            return {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 2244,
                            "Title": "Aspirin",
                            "ConnectivitySMILES": "CC(=O)Oc1ccccc1C(=O)O",
                        }
                    ]
                }
            }
        return {"molecules": []}


class MockSurechemblClient:
    """Track and mock SureChEMBL online chemical search calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.get_calls = 0

    def post_json(self, url: str, payload, timeout: float, debug: bool = False):
        self.calls += 1
        if url.startswith(SURECHEMBL_DOCUMENTS_FOR_STRUCTURES_URL):
            return {
                "status": "SUCCESS",
                "data": {
                    "documents": [
                        {
                            "chemical_id": "SCHEMBL-MOCK-ASPIRIN",
                            "patent_id": "MOCK-SURECHEMBL-PATENT-001",
                            "patent_title": "Mock SureChEMBL chemistry hit",
                            "publication_date": "2026-01-01",
                            "source_section": "claims",
                        }
                    ]
                },
            }
        return {
            "status": "SUCCESS",
            "data": {
                "results": [
                    {
                        "surechembl_id": "SCHEMBL-MOCK-ASPIRIN",
                        "patent_id": "MOCK-SURECHEMBL-PATENT-001",
                        "patent_title": "Mock SureChEMBL chemistry hit",
                        "patent_date": "2026-01-01",
                        "compound_name": "aspirin",
                        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                        "similarity": 1.0,
                        "source_section": "api_mock",
                    }
                ]
            },
        }

    def get_json(self, url: str, timeout: float, debug: bool = False):
        self.get_calls += 1
        return {"status": "SUCCESS", "data": {"results": []}}


def write_existing_public_lookup(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "molecule_id,canonical_smiles,inchi_key,valid_smiles,source_database,"
        "match_type,public_id,public_name,public_smiles,similarity,public_url,"
        "evidence_note,lookup_status,error_message\n"
        "aspirin,CC(=O)Oc1ccccc1C(=O)O,,True,PubChem,exact_inchikey,"
        "CID:2244,Aspirin,,1.000,,Exact public match.,match_found,\n",
        encoding="utf-8",
    )


def create_pipeline_inputs(root: Path) -> PipelinePaths:
    data_dir = root / "data"
    output_dir = root / "outputs"
    data_dir.mkdir()

    generated_smiles = data_dir / "generated.csv"
    generated_smiles.write_text(
        "molecule_id,smiles\n"
        "aspirin,CC(=O)Oc1ccccc1C(=O)O\n"
        "caffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O\n"
        "invalid,C1CC\n",
        encoding="utf-8",
    )
    references = data_dir / "references.csv"
    references.write_text(
        "reference_id,reference_name,smiles,reference_source,"
        "reference_source_id,evidence_note\n"
        "ref_aspirin,aspirin,CC(=O)Oc1ccccc1C(=O)O,public_demo,"
        "PUBLIC-001,Public-safe reference.\n"
        "ref_caffeine,caffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O,public_demo,"
        "PUBLIC-002,Public-safe reference.\n",
        encoding="utf-8",
    )
    text_evidence = data_dir / "evidence.csv"
    text_evidence.write_text(
        "evidence_id,molecule_id,source_type,title,text\n"
        "evidence_1,aspirin,synthetic,Demo evidence,Public-safe patent evidence\n"
        "evidence_2,caffeine,synthetic,Demo evidence,Public-safe molecule evidence\n"
        "evidence_3,invalid,synthetic,Missing evidence,\n",
        encoding="utf-8",
    )
    surechembl = data_dir / "surechembl.csv"
    surechembl.write_text(
        "surechembl_id,patent_id,patent_title,patent_date,compound_name,"
        "smiles,source_section,evidence_note\n"
        "SC-ASP,DEMO-PATENT-ASPIRIN-001,Demo aspirin chemistry record,"
        "2026-01-01,aspirin,CC(=O)Oc1ccccc1C(=O)O,claims_demo,"
        "Public-safe demonstration patent-associated compound evidence.\n",
        encoding="utf-8",
    )

    return PipelinePaths(
        generated_smiles=generated_smiles,
        references=references,
        text_evidence=text_evidence,
        surechembl_compounds=surechembl,
        target_profile_input=Path("data/demo_target/target_profile.csv"),
        target_profile=output_dir / "target_profile.csv",
        docking_results_normalized=output_dir / "docking_results_normalized.csv",
        structural_properties=output_dir / "structural_properties.csv",
        structural_prioritization_inputs=output_dir / "structural_prioritization_inputs.csv",
        standardized=output_dir / "standardized.csv",
        descriptors=output_dir / "descriptors.csv",
        admet_predictions=output_dir / "admet_predictions.csv",
        admet_summary=output_dir / "admet_summary.csv",
        similarity=output_dir / "similarity.csv",
        similarity_top_hits=output_dir / "similarity_top_hits.csv",
        text_nlp=output_dir / "text_nlp.csv",
        biomedical_evidence=output_dir / "biomedical_evidence.csv",
        patent_evidence_embeddings=output_dir / "patent_evidence_embeddings.csv",
        public_lookup=output_dir / "public_lookup.csv",
        surechembl_lookup=output_dir / "surechembl_lookup.csv",
        chemberta_embeddings=output_dir / "chemberta_embeddings.csv",
        visualization_coordinates=output_dir / "visualization_coordinates.csv",
        prioritized=output_dir / "prioritized.csv",
        biopharma_positioning=output_dir / "biopharma_positioning.csv",
        evidence_readiness=output_dir / "evidence_readiness.csv",
        mock_rwe_cohort_summary=output_dir / "mock_rwe_cohort_summary.csv",
        trial_endpoint_map=output_dir / "trial_endpoint_map.csv",
        biopharma_summary_report=output_dir / "biopharma_summary_report.md",
    )


def create_non_demo_inputs(root: Path) -> PipelinePaths:
    data_dir = root / "data"
    output_dir = root / "outputs"
    data_dir.mkdir()

    generated_smiles = data_dir / "generated.csv"
    generated_smiles.write_text(
        "molecule_id,smiles\n"
        "mol_a,CCO\n"
        "mol_b,CCN\n",
        encoding="utf-8",
    )
    references = data_dir / "references.csv"
    references.write_text(
        "reference_id,reference_name,smiles,reference_source,"
        "reference_source_id,evidence_note\n"
        "ref_ethanol,ethanol,CCO,public_demo,PUBLIC-001,Public-safe reference.\n",
        encoding="utf-8",
    )
    text_evidence = data_dir / "evidence.csv"
    text_evidence.write_text(
        "evidence_id,molecule_id,source_type,title,text\n",
        encoding="utf-8",
    )
    custom_surechembl = data_dir / "custom_surechembl.csv"
    custom_surechembl.write_text(
        "surechembl_id,patent_id,patent_title,patent_date,compound_name,"
        "smiles,source_section,evidence_note\n"
        "SC-CUSTOM-001,CUSTOM-PATENT-001,Custom local chemistry record,"
        "2026-01-01,custom_ethanol,CCO,claims,Custom local evidence.\n",
        encoding="utf-8",
    )

    return PipelinePaths(
        generated_smiles=generated_smiles,
        references=references,
        text_evidence=text_evidence,
        surechembl_compounds=custom_surechembl,
        target_profile_input=Path("data/demo_target/target_profile.csv"),
        target_profile=output_dir / "target_profile.csv",
        docking_results_normalized=output_dir / "docking_results_normalized.csv",
        structural_properties=output_dir / "structural_properties.csv",
        structural_prioritization_inputs=output_dir / "structural_prioritization_inputs.csv",
        standardized=output_dir / "standardized.csv",
        descriptors=output_dir / "descriptors.csv",
        admet_predictions=output_dir / "admet_predictions.csv",
        admet_summary=output_dir / "admet_summary.csv",
        similarity=output_dir / "similarity.csv",
        similarity_top_hits=output_dir / "similarity_top_hits.csv",
        text_nlp=output_dir / "text_nlp.csv",
        biomedical_evidence=output_dir / "biomedical_evidence.csv",
        patent_evidence_embeddings=output_dir / "patent_evidence_embeddings.csv",
        public_lookup=output_dir / "public_lookup.csv",
        surechembl_lookup=output_dir / "surechembl_lookup.csv",
        chemberta_embeddings=output_dir / "chemberta_embeddings.csv",
        visualization_coordinates=output_dir / "visualization_coordinates.csv",
        prioritized=output_dir / "prioritized.csv",
        biopharma_positioning=output_dir / "biopharma_positioning.csv",
        evidence_readiness=output_dir / "evidence_readiness.csv",
        mock_rwe_cohort_summary=output_dir / "mock_rwe_cohort_summary.csv",
        trial_endpoint_map=output_dir / "trial_endpoint_map.csv",
        biopharma_summary_report=output_dir / "biopharma_summary_report.md",
    )


def read_tree_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in root.rglob("*")
        if path.suffix.lower() in {".csv", ".md"}
    )


def test_pipeline_creates_expected_outputs(tmp_path: Path) -> None:
    paths = create_pipeline_inputs(tmp_path)
    lookup_client = MockLookupClient()
    surechembl_client = MockSurechemblClient()

    final_path = run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        lookup_client=lookup_client,
        surechembl_client=surechembl_client,
    )

    expected_outputs = (
        paths.standardized,
        paths.prioritized.parent / "chemical_identity.csv",
        paths.public_lookup,
        paths.surechembl_lookup,
        paths.target_profile,
        paths.structural_properties,
        paths.structural_prioritization_inputs,
        paths.descriptors,
        paths.admet_predictions,
        paths.admet_summary,
        paths.similarity,
        paths.similarity_top_hits,
        paths.prioritized.parent / "compound_context.csv",
        paths.text_nlp,
        paths.biomedical_evidence,
        paths.patent_evidence_embeddings,
        paths.prioritized,
        paths.biopharma_positioning,
        paths.evidence_readiness,
        paths.mock_rwe_cohort_summary,
        paths.trial_endpoint_map,
        paths.biopharma_summary_report,
    )
    assert final_path == paths.prioritized
    assert all(path.exists() for path in expected_outputs)

    with paths.prioritized.open(
        "r", encoding="utf-8", newline=""
    ) as output_file:
        rows = list(csv.DictReader(output_file))
        fieldnames = output_file

    assert "prioritization_score_with_nlp" in rows[0]
    assert "known_public_match" in rows[0]
    assert "ip_potential_category" in rows[0]
    assert "surechembl_signal_category" in rows[0]
    assert "chemical_identity_status" in rows[0]
    assert "chemical_identity_lookup_status" in rows[0]
    assert "context_status" in rows[0]
    assert "target_id" in rows[0]
    assert "docking_priority_label" in rows[0]
    assert rows[0]["docking_priority_label"] == "docking_unavailable"
    assert lookup_client.calls == 0
    assert surechembl_client.calls == 0
    assert not paths.chemberta_embeddings.exists()
    assert not paths.visualization_coordinates.exists()

    with paths.text_nlp.open(
        "r", encoding="utf-8", newline=""
    ) as nlp_file:
        nlp_rows = list(csv.DictReader(nlp_file))
    assert nlp_rows
    assert {row["molecule_id"] for row in nlp_rows} == {
        "aspirin",
        "caffeine",
    }
    assert all(row["nlp_status"] == "available" for row in nlp_rows)
    assert all(row["similarity_score"] for row in nlp_rows)
    assert {row["nlp_status"] for row in rows} == {"available", "no_match"}

    with paths.public_lookup.open(
        "r", encoding="utf-8", newline=""
    ) as lookup_file:
        lookup_rows = list(csv.DictReader(lookup_file))
    assert lookup_rows[0]["lookup_status"] == "offline"
    with paths.surechembl_lookup.open(
        "r", encoding="utf-8", newline=""
    ) as surechembl_file:
        surechembl_rows = list(csv.DictReader(surechembl_file))
    assert surechembl_rows[0]["lookup_status"] == "match_found"
    invalid_rows = [row for row in rows if row["molecule_id"] == "invalid"]
    assert len(invalid_rows) == 1
    assert invalid_rows[0]["valid_smiles"] == "False"
    assert (
        invalid_rows[0]["prioritization_category_with_nlp"]
        == "deprioritized"
    )


def test_pipeline_reports_logical_file_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = create_pipeline_inputs(tmp_path)

    run_pipeline(paths, nlp_model=MockSentenceModel())

    output = capsys.readouterr().out
    markers = [
        "Step 1/9: Standardizing and validating generated SMILES",
        "Step 2/9: Identifying exact public chemical names",
        "Step 3/9: Running public database and SureChEMBL lookups",
        "Step 4/9: Calculating RDKit properties and reference similarity",
        "Step 5/9: Generating ChemBERTa chemical-space outputs",
        "Step 6/9: Building biomedical context and scoring biomedical evidence",
        "Step 7/9: Scoring patent/IP-context evidence",
        "Step 8/9: Calculating final prioritization",
        "Step 9/9: Compound report generation skipped",
    ]
    positions = [output.index(marker) for marker in markers]
    assert positions == sorted(positions)


def test_prioritization_contains_status_from_all_prior_steps(
    tmp_path: Path,
) -> None:
    paths = create_non_demo_inputs(tmp_path)

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        skip_surechembl=True,
        use_chemberta=False,
    )

    with paths.prioritized.open(
        "r", encoding="utf-8", newline=""
    ) as prioritized_file:
        row = next(csv.DictReader(prioritized_file))
    assert row["chemical_identity_status"]
    assert row["chemical_identity_lookup_status"] == "offline"
    assert row["public_lookup_status"] == "offline"
    assert row["surechembl_query_status"] == "not_run"
    assert row["chemberta_status"] == "not_run"
    assert row["context_status"]
    assert row["nlp_status"] == "not_run"


def test_missing_input_file_gives_clear_error(tmp_path: Path) -> None:
    paths = create_pipeline_inputs(tmp_path)
    paths.text_evidence.unlink()

    with pytest.raises(
        FileNotFoundError, match="Missing required pipeline input file"
    ):
        run_pipeline(paths, nlp_model=MockSentenceModel())

    assert not paths.standardized.exists()


def test_existing_public_lookup_is_preserved_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = create_pipeline_inputs(tmp_path)
    write_existing_public_lookup(paths.public_lookup)
    before = paths.public_lookup.read_text(encoding="utf-8")
    lookup_client = MockLookupClient()

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        lookup_client=lookup_client,
    )

    assert paths.public_lookup.read_text(encoding="utf-8") == before
    assert lookup_client.calls == 0
    assert "Using existing public lookup file." in capsys.readouterr().out
    with paths.prioritized.open(
        "r", encoding="utf-8", newline=""
    ) as prioritized_file:
        rows = list(csv.DictReader(prioritized_file))
    assert rows[0]["known_public_match"] == "True"
    assert rows[0]["novelty_flag"] == "known_public_compound"
    assert rows[0]["ip_potential_category"] == "low_ip_potential_signal"


def test_offline_placeholder_generated_only_when_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = create_pipeline_inputs(tmp_path)

    run_pipeline(paths, nlp_model=MockSentenceModel())

    assert paths.public_lookup.exists()
    output = capsys.readouterr().out
    assert "No public lookup file found; generating offline placeholder lookup." in output


def test_refresh_public_lookup_overwrites_existing_file(tmp_path: Path) -> None:
    paths = create_pipeline_inputs(tmp_path)
    write_existing_public_lookup(paths.public_lookup)

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        refresh_public_lookup=True,
    )

    with paths.public_lookup.open(
        "r", encoding="utf-8", newline=""
    ) as lookup_file:
        rows = list(csv.DictReader(lookup_file))
    assert rows[0]["lookup_status"] == "offline"
    assert rows[0]["source_database"] == "offline"


def test_online_lookup_only_runs_when_enabled(tmp_path: Path) -> None:
    paths = create_pipeline_inputs(tmp_path)
    lookup_client = MockLookupClient()

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        online_lookup=True,
        max_molecules=1,
        refresh_public_lookup=True,
        lookup_client=lookup_client,
    )

    assert lookup_client.calls == 2
    with paths.public_lookup.open(
        "r", encoding="utf-8", newline=""
    ) as lookup_file:
        rows = list(csv.DictReader(lookup_file))
    assert rows[0]["lookup_status"] == "match_found"
    not_queried = [
        row for row in rows if row["molecule_id"] == "caffeine"
    ]
    assert not_queried
    assert {row["lookup_status"] for row in not_queried} == {"not_queried"}


def test_online_surechembl_only_runs_when_enabled(tmp_path: Path) -> None:
    paths = create_pipeline_inputs(tmp_path)
    surechembl_client = MockSurechemblClient()

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        online_surechembl=True,
        max_molecules=1,
        surechembl_client=surechembl_client,
    )

    assert surechembl_client.calls == 2
    with paths.surechembl_lookup.open(
        "r", encoding="utf-8", newline=""
    ) as surechembl_file:
        rows = list(csv.DictReader(surechembl_file))
    assert rows[0]["surechembl_id"] == "SCHEMBL-MOCK-ASPIRIN"


def test_online_surechembl_report_includes_api_evidence(tmp_path: Path) -> None:
    paths = create_pipeline_inputs(tmp_path)
    write_existing_public_lookup(paths.public_lookup)

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        online_surechembl=True,
        max_molecules=1,
        report_molecule="aspirin",
        surechembl_client=MockSurechemblClient(),
    )

    report = (
        paths.prioritized.parent
        / "reports"
        / "compound_intelligence_report_aspirin.md"
    )
    content = report.read_text(encoding="utf-8")
    assert "SureChEMBL Structure Evidence" in content
    assert "MOCK-SURECHEMBL-PATENT-001" in content
    assert "PatentsView" not in content
    assert "USPTO" not in content


def test_optional_report_generation(tmp_path: Path) -> None:
    paths = create_pipeline_inputs(tmp_path)
    write_existing_public_lookup(paths.public_lookup)

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        report_molecule="aspirin",
    )

    report = (
        paths.prioritized.parent
        / "reports"
        / "compound_intelligence_report_aspirin.md"
    )
    assert report.exists()
    content = report.read_text(encoding="utf-8")
    assert "# Compound Intelligence Report: aspirin" in content
    assert "Known public match: True" in content
    assert "Evidence completeness" in content
    assert "Evidence contribution summary" in content
    assert "Directly contributes property, QED, and Lipinski score components." in content
    assert "Adjusts the base prioritization score only when" in content
    assert "| PubChem status |" in content
    assert "SureChEMBL Structure Evidence" in content
    assert "Patent Evidence Summary" not in content
    assert "PatentsView" not in content
    assert "USPTO" not in content


def test_custom_cli_inputs_outputs_and_report_top_n(tmp_path: Path) -> None:
    paths = create_pipeline_inputs(tmp_path)
    output_dir = tmp_path / "custom_outputs"
    report_dir = output_dir / "custom_reports"

    exit_code = main(
        [
            "--input",
            str(paths.generated_smiles),
            "--references",
            str(paths.references),
            "--text-evidence",
            str(paths.text_evidence),
            "--output-dir",
            str(output_dir),
            "--report-top-n",
            "2",
            "--report-dir",
            str(report_dir),
        ],
        nlp_model=MockSentenceModel(),
    )

    assert exit_code == 0
    expected_outputs = (
        "standardized.csv",
        "descriptors.csv",
        "admet_predictions.csv",
        "admet_summary.csv",
        "similarity.csv",
        "similarity_top_hits.csv",
        "compound_context.csv",
        "text_nlp.csv",
        "biomedical_evidence.csv",
        "patent_evidence_embeddings.csv",
        "public_lookup.csv",
        "surechembl_evidence.csv",
        "prioritization_results.csv",
        "biopharma_positioning.csv",
        "evidence_readiness.csv",
        "mock_rwe_cohort_summary.csv",
        "trial_endpoint_map.csv",
        "biopharma_summary_report.md",
    )
    assert all((output_dir / name).exists() for name in expected_outputs)
    generated_names = [path.name for path in output_dir.glob("*")]
    assert not any("demo" in name.lower() for name in generated_names)
    assert not (output_dir / "patent_search_demo.csv").exists()
    assert report_dir.exists()
    reports = sorted(report_dir.glob("compound_intelligence_report_*.md"))
    assert len(reports) == 2

    with (output_dir / "prioritization_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as output_file:
        rows = list(csv.DictReader(output_file))
    molecule_ids = {row["molecule_id"] for row in rows}
    assert {"aspirin", "caffeine", "invalid"} <= molecule_ids


def test_report_only_fully_analyzed_excludes_not_queried(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = create_pipeline_inputs(tmp_path)

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        online_lookup=True,
        refresh_public_lookup=True,
        online_surechembl=True,
        max_molecules=1,
        report_top_n=2,
        report_only_fully_analyzed=True,
        lookup_client=MockLookupClient(),
        surechembl_client=MockSurechemblClient(),
    )

    output = capsys.readouterr().out
    assert "fewer fully analyzed molecules are available" in output
    reports = sorted(
        (paths.prioritized.parent / "reports").glob(
            "compound_intelligence_report_*.md"
        )
    )
    assert [report.name for report in reports] == [
        "compound_intelligence_report_aspirin.md"
    ]
    content = reports[0].read_text(encoding="utf-8")
    assert "Evidence completeness" in content
    assert "not checked because of max_molecules" not in content


def test_default_report_top_n_can_include_not_queried_with_visible_status(
    tmp_path: Path,
) -> None:
    paths = create_pipeline_inputs(tmp_path)

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        online_lookup=True,
        refresh_public_lookup=True,
        max_molecules=1,
        report_top_n=2,
        lookup_client=MockLookupClient(),
    )

    reports = sorted(
        (paths.prioritized.parent / "reports").glob(
            "compound_intelligence_report_*.md"
        )
    )
    text = "\n".join(report.read_text(encoding="utf-8") for report in reports)
    assert "Evidence completeness" in text
    assert "not_queried" in text
    assert "not checked because of max_molecules or workflow settings" in text


def test_skip_surechembl_prevents_demo_terms_in_reports(tmp_path: Path) -> None:
    paths = create_non_demo_inputs(tmp_path)
    output_dir = tmp_path / "clean_outputs"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True)
    stale_report = report_dir / "compound_intelligence_report_stale.md"
    stale_report.write_text("old DEMO-PATENT aspirin", encoding="utf-8")

    exit_code = main(
        [
            "--input",
            str(paths.generated_smiles),
            "--references",
            str(paths.references),
            "--text-evidence",
            str(paths.text_evidence),
            "--output-dir",
            str(output_dir),
            "--skip-surechembl",
            "--report-top-n",
            "1",
            "--report-dir",
            str(report_dir),
            "--clean-report-dir",
        ],
        nlp_model=MockSentenceModel(),
    )

    assert exit_code == 0
    assert not stale_report.exists()
    report_text = read_tree_text(report_dir)
    assert "SureChEMBL structure evidence was not run for this workflow." in report_text
    for term in (
        "aspirin",
        "caffeine",
        "benzene",
        "ibuprofen",
        "acetaminophen",
        "DEMO-PATENT",
    ):
        assert term.lower() not in report_text.lower()


def test_custom_cli_skips_demo_surechembl_by_default(tmp_path: Path) -> None:
    paths = create_non_demo_inputs(tmp_path)
    output_dir = tmp_path / "custom_outputs"

    main(
        [
            "--input",
            str(paths.generated_smiles),
            "--references",
            str(paths.references),
            "--text-evidence",
            str(paths.text_evidence),
            "--output-dir",
            str(output_dir),
            "--report-top-n",
            "1",
        ],
        nlp_model=MockSentenceModel(),
    )

    text = read_tree_text(output_dir)
    assert "SureChEMBL structure evidence was not run for this workflow." in text
    assert "DEMO-PATENT" not in text


def test_use_demo_surechembl_keeps_demo_behavior_for_custom_run(
    tmp_path: Path,
) -> None:
    paths = create_non_demo_inputs(tmp_path)
    output_dir = tmp_path / "demo_surechembl_outputs"

    main(
        [
            "--input",
            str(paths.generated_smiles),
            "--references",
            str(paths.references),
            "--text-evidence",
            str(paths.text_evidence),
            "--output-dir",
            str(output_dir),
            "--use-demo-surechembl",
            "--report-top-n",
            "1",
        ],
        nlp_model=MockSentenceModel(),
    )

    text = read_tree_text(output_dir)
    assert "DEMO-PATENT" in text
    assert "aspirin" in text.lower()


def test_custom_surechembl_compounds_file_is_used(tmp_path: Path) -> None:
    paths = create_non_demo_inputs(tmp_path)
    output_dir = tmp_path / "custom_surechembl_outputs"
    custom_file = paths.surechembl_compounds

    main(
        [
            "--input",
            str(paths.generated_smiles),
            "--references",
            str(paths.references),
            "--text-evidence",
            str(paths.text_evidence),
            "--output-dir",
            str(output_dir),
            "--surechembl-compounds",
            str(custom_file),
            "--report-top-n",
            "1",
        ],
        nlp_model=MockSentenceModel(),
    )

    text = read_tree_text(output_dir)
    assert "custom_ethanol" in text
    assert "CUSTOM-PATENT-001" in text
    assert "DEMO-PATENT" not in text


def test_pipeline_creates_chemberta_outputs_when_requested(tmp_path: Path) -> None:
    paths = create_non_demo_inputs(tmp_path)
    embedder = MockChembertaEmbedder()

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        skip_surechembl=True,
        use_chemberta=True,
        chemberta_embedder=embedder,
        report_molecule="mol_a",
    )

    assert embedder.calls == 2
    assert paths.chemberta_embeddings.exists()
    assert paths.visualization_coordinates.exists()
    with paths.visualization_coordinates.open(
        "r", encoding="utf-8", newline=""
    ) as visualization_file:
        visualization_rows = list(csv.DictReader(visualization_file))
    assert {row["source_type"] for row in visualization_rows} == {
        "generated",
        "reference",
    }
    assert len(visualization_rows) == 3
    assert {
        row["coordinate_method"]
        for row in visualization_rows
        if row["coordinate_method"] != "not_available"
    }
    generated_rows = [
        row for row in visualization_rows if row["source_type"] == "generated"
    ]
    assert all(row["nearest_reference_id"] for row in generated_rows)
    assert all(row["nearest_reference_name"] for row in generated_rows)
    assert all(row["nearest_reference_similarity"] for row in generated_rows)
    assert next(
        row for row in generated_rows if row["molecule_id"] == "mol_a"
    )["nearest_reference_interpretation"] == "high_similarity"
    with paths.prioritized.open(
        "r", encoding="utf-8", newline=""
    ) as prioritized_file:
        rows = list(csv.DictReader(prioritized_file))
    assert rows[0]["chemberta_model"] == "mock-chemberta"
    assert rows[0]["chemberta_embedding_available"] == "True"

    report = (
        paths.prioritized.parent
        / "reports"
        / "compound_intelligence_report_mol_a.md"
    )
    content = report.read_text(encoding="utf-8")
    assert "ChemBERTa learned chemical-space representation" in content


def test_invalid_smiles_has_chemberta_not_available_without_failure(
    tmp_path: Path,
) -> None:
    paths = create_pipeline_inputs(tmp_path)

    run_pipeline(
        paths,
        nlp_model=MockSentenceModel(),
        skip_surechembl=True,
        use_chemberta=True,
        chemberta_embedder=MockChembertaEmbedder(),
    )

    with paths.prioritized.open(
        "r", encoding="utf-8", newline=""
    ) as prioritized_file:
        rows = list(csv.DictReader(prioritized_file))
    invalid = next(row for row in rows if row["molecule_id"] == "invalid")
    assert invalid["chemberta_status"] == "not_available"
