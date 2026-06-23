from pathlib import Path
from datetime import datetime
from io import BytesIO
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app


def test_app_imports_without_crashing() -> None:
    assert hasattr(app, "run_app")
    assert hasattr(app, "load_output_directory")


def test_app_uses_molecule_design_title_and_subtitle() -> None:
    app_test = AppTest.from_file("app.py").run(timeout=10)

    assert app.APP_TITLE == "AI Molecule Design Assistant"
    assert app.APP_TITLE in [item.value for item in app_test.title]
    assert not hasattr(app, "APP_SUBTITLE")


def test_app_startup_does_not_auto_load_output_folder() -> None:
    app_test = AppTest.from_file("app.py").run(timeout=10)

    assert not app_test.exception
    assert not any(
        heading.value in app.WORKFLOW_STEP_NAMES for heading in app_test.header
    )
    assert not app_test.dataframe


def test_about_workflow_and_start_guidance_exist() -> None:
    app_test = AppTest.from_file("app.py").run(timeout=10)

    assert app.START_GUIDANCE in [item.value for item in app_test.info]
    button_labels = [button.label for button in app_test.button]
    assert "Run guided example workflow" in button_labels
    assert "Run analysis" in button_labels
    assert len(app.ABOUT_WORKFLOW_SECTIONS) == 7
    about_text = " ".join(
        f"{heading} {explanation}"
        for heading, explanation in app.ABOUT_WORKFLOW_SECTIONS
    )
    for source_name in (
        "PubChem PUG-REST",
        "ChEMBL web services",
        "SureChEMBL",
        "RDKit documentation",
        "Lipinski framework",
        "QED",
        "ChemBERTa",
        "UMAP",
        "Sentence Transformers documentation",
    ):
        assert source_name in about_text


def test_workflow_step_names_exist() -> None:
    assert app.WORKFLOW_STEP_NAMES == (
        "Step 1: Load and validate SMILES",
        "Step 2: Chemical identity",
        "Step 3: Public database lookup",
        "Step 4: RDKit molecular properties",
        "Step 5: ChemBERTa chemical space",
        "Step 6: Text evidence and biomedical context",
        "Step 7: Final prioritization",
        "Step 8: Reports",
    )
    assert app.FINAL_RANKING_EXPLANATION == (
        "Final ranking combines evidence from chemical identity, public lookup, "
        "RDKit descriptors, ChemBERTa embeddings, text evidence, and biomedical context."
    )


def test_demo_results_require_explicit_action() -> None:
    app_test = AppTest.from_file("app.py").run(timeout=10)

    assert any(
        button.label == "Run guided example workflow"
        for button in app_test.button
    )
    assert not any(
        heading.value in app.WORKFLOW_STEP_NAMES for heading in app_test.header
    )


def test_demo_results_load_after_explicit_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "app_runs" / "public_demo" / "outputs"
    output.mkdir(parents=True)
    calls = []
    paths = app.build_paths(
        input_path=app.DEMO_INPUT,
        references_path=app.DEMO_REFERENCES,
        text_evidence_path=app.DEMO_TEXT_EVIDENCE,
        output_dir=output,
    )
    fake_streamlit = SimpleNamespace(
        session_state={},
        header=lambda *args, **kwargs: None,
        write=lambda *args, **kwargs: None,
        markdown=lambda *args, **kwargs: None,
        code=lambda *args, **kwargs: None,
        button=lambda *args, **kwargs: True,
        error=lambda *args, **kwargs: None,
        success=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(app, "st", fake_streamlit)
    monkeypatch.setattr(
        app,
        "create_public_demo_workflow",
        lambda: calls.append("start") or paths,
    )

    result = app.render_public_demo_choice()

    assert calls == ["start"]
    assert result == output
    assert fake_streamlit.session_state["active_output_dir"] == str(output)
    assert fake_streamlit.session_state["completed_workflow_steps"] == []


def test_existing_results_require_explicit_action(tmp_path: Path) -> None:
    app_test = AppTest.from_file("app.py").run(timeout=10)

    assert any(
        text_input.label == "Output directory"
        for text_input in app_test.text_input
    )
    assert any(button.label == "Load results" for button in app_test.button)
    assert not any(
        heading.value in app.WORKFLOW_STEP_NAMES for heading in app_test.header
    )

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "prioritization_results.csv").write_text(
        "molecule_id,valid_smiles,prioritization_score,known_public_match\n"
        "mol_a,True,0.8,False\n",
        encoding="utf-8",
    )
    output_input = next(
        item for item in app_test.text_input if item.label == "Output directory"
    )
    output_input.set_value(str(output_dir))
    load_button = next(
        button for button in app_test.button if button.label == "Load results"
    )
    load_button.click().run(timeout=10)

    assert any(
        heading.value == "Step 1: Load and validate SMILES"
        for heading in app_test.header
    )


def test_public_demo_uses_bundled_inputs_and_new_output_folder(
) -> None:
    result = app.create_public_demo_workflow(
        timestamp=datetime(2026, 6, 19, 15, 30, 0)
    )

    assert result.generated_smiles == app.DEMO_INPUT
    assert result.references == app.DEMO_REFERENCES
    assert result.text_evidence == app.DEMO_TEXT_EVIDENCE
    assert result.prioritized.parent == Path(
        "app_runs/public_demo_20260619_153000/outputs"
    )


def test_public_demo_step_one_runs_only_standardization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = app.build_paths(
        input_path=app.DEMO_INPUT,
        references_path=app.DEMO_REFERENCES,
        text_evidence_path=app.DEMO_TEXT_EVIDENCE,
        output_dir=tmp_path / "outputs",
    )
    calls = []
    monkeypatch.setattr(
        app,
        "standardize_csv",
        lambda input_path, output_path: calls.append(
            (input_path, output_path)
        ),
    )
    monkeypatch.setattr(
        app,
        "chemical_identity_csv",
        lambda *args, **kwargs: pytest.fail("Step 2 must not run"),
    )

    app.run_public_demo_step(1, paths)

    assert calls == [(app.DEMO_INPUT, paths.standardized)]


def test_public_demo_opens_step_one_before_running_calculation() -> None:
    app_test = AppTest.from_file("app.py").run(timeout=10)
    demo_button = next(
        button
        for button in app_test.button
        if button.label == "Run guided example workflow"
    )
    demo_button.click().run(timeout=10)

    assert any(
        heading.value == "Step 1: Load and validate SMILES"
        for heading in app_test.header
    )
    assert any(
        button.label == "Run Step 1 on public example"
        for button in app_test.button
    )
    markdown_values = [item.value for item in app_test.markdown]
    assert "#### What this step calculates" in markdown_values
    assert "#### Why we run it" in markdown_values
    assert "#### What you will get" in markdown_values
    assert not app_test.dataframe


def test_display_label_mapping_uses_readable_names() -> None:
    assert (
        app.display_label("prioritization_score_with_nlp")
        == "Research-prioritization score"
    )
    assert app.display_label("tanimoto_similarity") == "Best reference similarity"
    assert app.display_label("nlp_status") == "Text-evidence status"
    assert app.display_label("custom_internal_name") == "Custom Internal Name"
    assert (
        app.display_label("druglikeness_category")
        == "Drug-likeness category"
    )


def test_druglikeness_helpers_return_readable_labels_and_counts() -> None:
    descriptors = pd.DataFrame(
        {
            "molecule_id": ["a", "b", "c", "d"],
            "druglikeness_category": [
                "favorable",
                "borderline",
                "unfavorable",
                "invalid",
            ],
            "mw_status": ["favorable", "borderline", "unfavorable", "invalid"],
            "logp_status": ["favorable"] * 4,
            "tpsa_status": ["favorable"] * 4,
            "qed_status": ["favorable"] * 4,
            "lipinski_status": ["favorable"] * 4,
        }
    )

    assert app.druglikeness_counts(descriptors) == {
        "favorable": 1,
        "borderline": 1,
        "unfavorable": 1,
        "invalid": 1,
    }
    matrix = app.druglikeness_status_matrix(descriptors)
    assert matrix.columns.tolist() == [
        "Molecule ID",
        "MW",
        "LogP",
        "TPSA",
        "QED",
        "Lipinski",
    ]
    assert matrix.loc[0, "MW"] == "✅ favorable"
    assert matrix.loc[1, "MW"] == "⚠️ borderline"
    assert matrix.loc[2, "MW"] == "❌ unfavorable"


def test_table_and_plot_helpers_hide_internal_column_names() -> None:
    df = pd.DataFrame(
        {
            "molecule_id": ["mol_a"],
            "prioritization_score_with_nlp": [0.9],
            "best_reference_name": ["Reference A"],
        }
    )

    displayed = app.display_dataframe(df)
    plot_labels = app.display_labels(df.columns)

    assert "prioritization_score_with_nlp" not in displayed.columns
    assert (
        plot_labels["prioritization_score_with_nlp"]
        == "Research-prioritization score"
    )
    assert "prioritization_score_with_nlp" not in plot_labels.values()


def test_shared_status_colors_follow_workflow_semantics() -> None:
    assert app.molecule_status_color("valid") == "green"
    assert app.molecule_status_color("available") == "green"
    assert app.molecule_status_color("borderline") == "yellow"
    assert app.molecule_status_color("partial") == "yellow"
    assert app.molecule_status_color("invalid") == "red"
    assert app.molecule_status_color("not_available") == "red"
    assert app.molecule_status_color("not_run") == "gray"


def test_validation_visualization_data_is_molecule_level() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["mol_a", "mol_b"],
            "smiles": ["CCO", "invalid"],
            "valid_smiles": [True, False],
        }
    )

    plot = app.validation_molecule_dataframe(frame)

    assert plot["molecule_id"].tolist() == ["mol_a", "mol_b"]
    assert plot["validation_status"].tolist() == ["Valid", "Invalid"]
    assert plot["status_color"].tolist() == ["green", "red"]
    assert plot["molecule_position"].tolist() == [1, 2]


def test_identity_visualization_data_keeps_readable_molecule_parameters() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["mol_a", "mol_b"],
            "identity_status": ["exact_public_identity", "no_public_identity"],
            "exact_public_name": ["Example", ""],
            "iupac_name": ["example-name", ""],
            "name_source": ["PubChem", ""],
            "identity_confidence": ["high", "low"],
        }
    )

    plot = app.identity_molecule_dataframe(frame)

    assert len(plot) == 2
    assert plot.loc[0, "exact_public_name"] == "Example"
    assert plot["status_color"].tolist() == ["green", "red"]


def test_public_evidence_visualization_returns_one_row_per_molecule() -> None:
    public_lookup = pd.DataFrame(
        {
            "molecule_id": ["mol_a", "mol_a", "mol_b"],
            "source_database": ["PubChem", "ChEMBL", "PubChem"],
            "lookup_status": ["match_found", "no_match", "not_queried"],
        }
    )
    surechembl = pd.DataFrame(
        {
            "molecule_id": ["mol_a"],
            "lookup_status": ["match_found"],
        }
    )

    evidence = app.public_evidence_molecule_dataframe(
        public_lookup, surechembl, ["mol_a", "mol_b"]
    ).set_index("molecule_id")

    assert evidence.index.tolist() == ["mol_a", "mol_b"]
    assert evidence.loc["mol_a", "pubchem_status"] == "match_found"
    assert evidence.loc["mol_a", "surechembl_query_status"] == "match_found"
    assert evidence.loc["mol_b", "chembl_status"] == "not_run"


def test_rdkit_visualization_data_contains_molecule_hover_parameters() -> None:
    descriptors = pd.DataFrame(
        {
            "molecule_id": ["mol_a"],
            "logp": ["2.1"],
            "qed": ["0.72"],
            "molecular_weight": ["315.4"],
            "tpsa": ["68.2"],
            "hbd": ["1"],
            "hba": ["4"],
            "rotatable_bonds": ["3"],
            "lipinski_pass": ["True"],
            "druglikeness_category": ["favorable"],
        }
    )

    plot = app.rdkit_molecule_dataframe(descriptors)

    assert plot.loc[0, "molecule_id"] == "mol_a"
    assert plot.loc[0, "qed"] == pytest.approx(0.72)
    assert plot.loc[0, "hbd"] == pytest.approx(1)
    assert plot.loc[0, "hba"] == pytest.approx(4)
    assert plot.loc[0, "rotatable_bonds"] == pytest.approx(3)
    assert plot.loc[0, "status_color"] == "green"


def test_text_evidence_visualization_summarizes_each_molecule() -> None:
    text_nlp = pd.DataFrame(
        {
            "molecule_id": ["mol_a", "mol_a"],
            "nlp_status": ["available", "available"],
            "title": ["Lower match", "Top match"],
            "max_relevance_score": [0.3, 0.8],
        }
    )

    plot = app.text_evidence_molecule_dataframe(text_nlp, ["mol_a", "mol_b"])
    rows = plot.set_index("molecule_id")

    assert rows.loc["mol_a", "top_evidence_title"] == "Top match"
    assert rows.loc["mol_a", "evidence_matches"] == 2
    assert rows.loc["mol_b", "nlp_status"] == "not_run"
    assert rows.loc["mol_b", "status_color"] == "gray"


def test_final_priority_visualization_uses_molecule_level_scores() -> None:
    prioritization = pd.DataFrame(
        {
            "molecule_id": ["mol_a", "mol_b"],
            "prioritization_score_with_nlp": ["0.91", "0.42"],
            "tanimoto_similarity": ["0.20", "0.70"],
            "prioritization_category_with_nlp": [
                "high_priority",
                "low_priority",
            ],
        }
    )

    plot = app.final_priority_molecule_dataframe(prioritization)

    assert plot["molecule_id"].tolist() == ["mol_a", "mol_b"]
    assert plot["design_category"].tolist() == ["high_priority", "low_priority"]
    assert plot["status_color"].tolist() == ["green", "red"]


def test_plot_selection_extracts_molecule_id() -> None:
    event = {"selection": {"points": [{"customdata": ["mol_a"]}]}}

    assert app.selected_molecule_from_plot_event(event) == "mol_a"
    assert app.selected_molecule_from_plot_event(None) == ""


def test_new_visualization_labels_do_not_expose_internal_names() -> None:
    columns = [
        "prioritization_score_with_nlp",
        "prioritization_category_with_nlp",
        "tanimoto_similarity",
        "surechembl_query_status",
        "rotatable_bonds",
    ]

    labels = app.display_labels(columns)

    assert set(labels.values()) == {
        "Research-prioritization score",
        "Final design category",
        "Best reference similarity",
        "SureChEMBL status",
        "Rotatable bonds",
    }
    assert not any(column in labels.values() for column in columns)


def test_graph_explanation_text_exists() -> None:
    assert app.CHEMICAL_SPACE_EXPLANATION == (
        "Each point is one molecule. Molecules close together have similar "
        "ChemBERTa molecular embeddings. Use this plot to identify clusters, "
        "outliers, and chemically distinct candidates."
    )
    assert "High score with low reference similarity" in (
        app.SCORE_SIMILARITY_EXPLANATION
    )
    assert "molecular weight, LogP, TPSA, and QED" in (
        app.PROPERTY_DISTRIBUTIONS_EXPLANATION
    )


def test_output_loading_handles_missing_optional_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "prioritization_results.csv").write_text(
        "molecule_id,prioritization_score,known_public_match\n"
        "mol_a,0.750,False\n",
        encoding="utf-8",
    )

    loaded = app.load_output_directory(output_dir)

    assert len(loaded.tables["prioritization"]) == 1
    assert loaded.tables["visualization"].empty
    assert loaded.tables["descriptors"].empty
    assert loaded.reports_dir == output_dir / "reports"
    assert loaded.images_dir == output_dir / "report_images"


def test_output_loading_preserves_prioritization_nlp_status(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "prioritization_results.csv").write_text(
        "molecule_id,nlp_status\n"
        "mol_a,not_run\n"
        "mol_b,not_run\n",
        encoding="utf-8",
    )
    (output_dir / "text_nlp.csv").write_text(
        "molecule_id,evidence_id,similarity_score,nlp_status\n"
        "mol_a,ev_1,0.750,available\n",
        encoding="utf-8",
    )

    loaded = app.load_output_directory(output_dir)
    statuses = loaded.tables["prioritization"].set_index("molecule_id")[
        "nlp_status"
    ]

    assert statuses["mol_a"] == "not_run"
    assert statuses["mol_b"] == "not_run"


def test_missing_text_nlp_file_produces_not_run_note(tmp_path: Path) -> None:
    path = tmp_path / "text_nlp.csv"

    assert app.nlp_output_note(path, pd.DataFrame()) == (
        "NLP was not run for this output folder because no text_nlp.csv file "
        "was found. Rerun the pipeline with --text-evidence to enable NLP "
        "evidence matching."
    )


def test_empty_text_nlp_file_produces_empty_output_note(
    tmp_path: Path,
) -> None:
    path = tmp_path / "text_nlp.csv"
    path.write_text("molecule_id,evidence_id\n", encoding="utf-8")

    assert app.nlp_output_note(path, pd.DataFrame()) == (
        "NLP output exists but contains no evidence matches."
    )


def test_nonempty_text_nlp_file_produces_ran_note(tmp_path: Path) -> None:
    path = tmp_path / "text_nlp.csv"
    path.write_text(
        "molecule_id,evidence_id\nmol_a,ev_1\n",
        encoding="utf-8",
    )

    assert app.nlp_output_note(
        path, pd.DataFrame({"molecule_id": ["mol_a"]})
    ) == "NLP evidence matching was run using molecule context and text evidence."


def test_expected_files_loaded_from_existing_quality_output_if_present() -> None:
    output_dir = Path("outputs/public_druglike_demo")
    if not output_dir.exists():
        return

    loaded = app.load_output_directory(output_dir)

    assert not loaded.tables["prioritization"].empty
    assert "molecule_id" in loaded.tables["prioritization"].columns
    assert not loaded.tables["descriptors"].empty
    if loaded.paths["visualization"].exists():
        assert not loaded.tables["visualization"].empty


def test_missing_visualization_coordinates_is_graceful(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "prioritization_results.csv").write_text(
        "molecule_id,prioritization_score,tanimoto_similarity,known_public_match\n"
        "mol_a,0.750,0.100,False\n",
        encoding="utf-8",
    )

    loaded = app.load_output_directory(output_dir)
    filtered = app.apply_filters(
        loaded.tables["prioritization"],
        min_score=0.7,
        known_public_match="All",
        status_filters={},
    )

    assert loaded.tables["visualization"].empty
    assert filtered["molecule_id"].tolist() == ["mol_a"]


def test_chemical_space_works_before_prioritization_exists() -> None:
    coordinates = pd.DataFrame(
        {
            "molecule_id": ["demo_001", "demo_002"],
            "x": ["0.1", "0.2"],
            "y": ["-0.3", "0.4"],
            "coordinate_method": ["pca", "pca"],
            "cluster_id": ["2", "0"],
        }
    )

    plot = app.chemical_space_dataframe(coordinates, pd.DataFrame())

    assert plot["molecule_id"].tolist() == ["demo_001", "demo_002"]
    assert plot["x"].tolist() == [0.1, 0.2]
    assert plot["y"].tolist() == [-0.3, 0.4]


def test_status_counts_and_column_order() -> None:
    df = pd.DataFrame(
        {
            "molecule_id": ["a", "b"],
            "pubchem_status": ["no_match", "not_queried"],
            "extra": [1, 2],
        }
    )

    assert app.status_counts(df, "pubchem_status") == {
        "no_match": 1,
        "not_queried": 1,
    }
    assert app.ordered_columns(df)[:2] == ["molecule_id", "pubchem_status"]


def test_summarize_run_returns_clear_cards() -> None:
    df = pd.DataFrame(
        {
            "molecule_id": ["a", "b", "c"],
            "valid_smiles": ["True", "False", "True"],
            "known_public_match": ["False", "True", "False"],
            "chemberta_status": ["available", "not_available", "available"],
        }
    )

    summary = app.summarize_run(df)

    assert summary == {
        "Total": 3,
        "Valid": 2,
        "Exact public matches": 1,
        "ChemBERTa": 2,
    }
    assert all(isinstance(value, int) for value in summary.values())

    not_run = app.summarize_run(df, public_lookup_exists=False)
    assert not_run["Exact public matches"] == "Not run"


def test_external_public_evidence_table_uses_count_columns() -> None:
    df = pd.DataFrame(
        {
            "pubchem_status": ["no_match", "not_queried", "match_found"],
            "chembl_status": ["lookup_error", "no_match", "not_queried"],
            "surechembl_query_status": ["no_match", "not_queried", "no_match"],
            "chemberta_status": ["available", "available", "not_available"],
            "nlp_status": ["not_run", "not_run", "available"],
        }
    )

    table = app.build_external_public_evidence_table(df)

    assert table.columns.tolist() == [
        "Source",
        "Hit",
        "No match",
        "Not queried",
        "Not run",
        "Error",
    ]
    assert table["Source"].tolist() == [
        "PubChem",
        "ChEMBL",
        "SureChEMBL",
    ]
    numeric_columns = [column for column in table.columns if column != "Source"]
    assert all(pd.api.types.is_integer_dtype(table[column]) for column in numeric_columns)
    assert "no_match" not in table[numeric_columns].to_string()
    pubchem = table[table["Source"] == "PubChem"].iloc[0]
    assert pubchem["Hit"] == 1
    assert pubchem["No match"] == 1
    assert pubchem["Not queried"] == 1
    assert pubchem["Not run"] == 0


def test_missing_public_evidence_columns_count_as_not_run() -> None:
    df = pd.DataFrame({"molecule_id": ["a", "b", "c"]})

    table = app.build_external_public_evidence_table(df)

    assert table["Not run"].tolist() == [3, 3, 3]
    assert table[["Hit", "No match", "Not queried", "Error"]].to_numpy().sum() == 0


def test_computed_analysis_table_counts_chemberta_as_available_not_hit() -> None:
    df = pd.DataFrame(
        {
            "chemberta_status": ["available", "available", "not_available"],
            "nlp_status": ["not_run", "not_run", "available"],
            "valid_smiles": ["True", "True", "True"],
        }
    )

    table = app.build_computed_analysis_status_table(df)

    assert table.columns.tolist() == [
        "Source",
        "Available",
        "Not run",
        "Not available",
        "Error",
    ]
    assert table["Source"].tolist() == [
        "ChemBERTa",
        "Text-evidence matching",
        "RDKit descriptors",
    ]
    chemberta = table[table["Source"] == "ChemBERTa"].iloc[0]
    assert "Hit" not in table.columns
    assert chemberta["Available"] == 2
    assert chemberta["Not available"] == 1
    nlp = table[table["Source"] == "Text-evidence matching"].iloc[0]
    assert nlp["Available"] == 1
    assert nlp["Not run"] == 2


def test_computed_analysis_table_counts_nlp_no_match_as_not_available() -> None:
    df = pd.DataFrame(
        {
            "chemberta_status": ["available", "not_available", "available"],
            "nlp_status": ["available", "no_match", "not_run"],
            "valid_smiles": ["True", "False", "True"],
        }
    )

    table = app.build_computed_analysis_status_table(df)
    nlp = table[table["Source"] == "Text-evidence matching"].iloc[0]

    assert nlp["Available"] == 1
    assert nlp["Not available"] == 1
    assert nlp["Not run"] == 1


def test_computed_analysis_nlp_counts_use_prioritization_status() -> None:
    df = pd.DataFrame(
        {
            "nlp_status": [
                "available",
                "available",
                "no_match",
                "not_available",
                "not_run",
                "error",
                "lookup_error",
            ],
            "valid_smiles": ["True"] * 7,
        }
    )

    table = app.build_computed_analysis_status_table(df)
    nlp = table[table["Source"] == "Text-evidence matching"].iloc[0]

    assert nlp["Available"] == 2
    assert nlp["Not available"] == 2
    assert nlp["Not run"] == 1
    assert nlp["Error"] == 2


def test_status_meaning_mapping() -> None:
    assert app.status_meaning("no_match") == "queried; no match found"
    assert app.status_meaning("hit") == "queried; match found"
    assert app.status_meaning("match_found") == "queried; match found"
    assert app.status_meaning("not_queried") == "not checked in this run"
    assert app.status_meaning("lookup_error") == "query failed"
    assert app.status_meaning("available") == "available"
    assert app.status_meaning("not_run") == "workflow step was skipped"


def test_readable_status_hides_machine_codes() -> None:
    assert app.readable_status("match_found") == "Match found"
    assert app.readable_status("lookup_error") == "Lookup error"
    assert app.readable_status("exact_public_identity") == "Exact public identity"
    assert app.readable_status("high_priority") == "High priority"
    assert app.readable_status("not_queried") == "Not queried"
    assert app.readable_status("not_run") == "Not run"
    assert "_" not in app.status_display("match_found")


def test_readable_ui_dataframe_hides_status_codes() -> None:
    frame = pd.DataFrame(
        {
            "identity_status": ["exact_public_identity"],
            "lookup_status": ["match_found"],
            "nlp_status": ["not_run"],
            "design_category": ["high_priority"],
            "molecule_id": ["mol_a"],
        }
    )

    readable = app.readable_ui_dataframe(frame)

    assert readable.loc[0, "identity_status"] == "Exact public identity"
    assert readable.loc[0, "lookup_status"] == "Match found"
    assert readable.loc[0, "nlp_status"] == "Not run"
    assert readable.loc[0, "design_category"] == "High priority"
    assert readable.loc[0, "molecule_id"] == "mol_a"
    assert "_" not in " ".join(
        readable.loc[
            0,
            ["identity_status", "lookup_status", "nlp_status", "design_category"],
        ].astype(str)
    )


def test_compact_detail_dataframe_removes_empty_fields() -> None:
    frame = pd.DataFrame(
        {
            "identity_status": ["no_public_identity"],
            "exact_public_name": [""],
            "preferred_name": [None],
            "identity_confidence": ["none"],
            "inchikey": ["ABC-DEF"],
            "lookup_status": ["not_queried"],
        }
    )

    compact = app.compact_detail_dataframe(frame)

    assert compact.columns.tolist() == [
        "identity_status",
        "inchikey",
        "lookup_status",
    ]


def test_artifact_display_name_hides_local_directories() -> None:
    path = Path(
        "app_runs/public_demo_20260622_153014/outputs/chemical_identity.csv"
    )

    assert app.artifact_display_name(path) == "chemical_identity.csv"
    assert app.artifact_display_name(
        r"app_runs\public_demo\outputs\surechembl_evidence.csv"
    ) == "surechembl_evidence.csv"
    assert app.artifact_display_name("uploaded SMILES CSV") == "uploaded SMILES CSV"


def test_evidence_completeness_rows_are_table_ready() -> None:
    rows = app.evidence_completeness_rows(
        {
            "pubchem_status": "no_match",
            "chembl_status": "hit",
            "surechembl_query_status": "not_queried",
            "chemberta_status": "available",
            "nlp_status": "not_run",
        }
    )

    assert rows.columns.tolist() == ["Evidence source", "Status", "Meaning"]
    assert rows["Evidence source"].tolist() == [
        "PubChem",
        "ChEMBL",
        "SureChEMBL",
        "ChemBERTa",
        "NLP",
    ]
    assert "{" not in rows.to_string()
    assert "queried; no match found" in rows["Meaning"].tolist()


def test_molecule_detail_rows_are_table_ready() -> None:
    rows = app.molecule_detail_rows(
        {
            "prioritization_score_with_nlp": 0.91,
            "best_reference_name": "Reference A",
            "tanimoto_similarity": 0.21,
            "known_public_match": "False",
            "pubchem_status": "no_match",
            "chembl_status": "lookup_error",
            "surechembl_query_status": "not_queried",
            "chemberta_status": "available",
            "nlp_status": "not_run",
        },
        "prioritization_score_with_nlp",
    )

    assert rows.columns.tolist() == ["Field", "Value"]
    assert "prioritization_score_with_nlp" not in rows["Field"].tolist()
    assert "Research-prioritization score" in rows["Field"].tolist()
    assert "Text-evidence status" in rows["Field"].tolist()
    assert "Reference similarity interpretation" in rows["Field"].tolist()
    assert "Weak RDKit fingerprint similarity" in rows["Value"].astype(str).to_string()
    assert all(str(dtype) == "string" for dtype in rows.dtypes)


def test_missing_molecule_image_message(tmp_path: Path) -> None:
    image_path = tmp_path / "missing.png"

    assert (
        app.molecule_image_message(image_path)
        == "2D structure image is not available for this molecule."
    )


def test_molecule_structure_image_is_generated_for_valid_smiles() -> None:
    image = app.molecule_structure_image("CCO")

    assert image is not None
    assert image.size == (420, 320)


def test_molecule_structure_image_rejects_invalid_smiles() -> None:
    assert app.molecule_structure_image("not-a-smiles") is None
    assert app.molecule_structure_image("") is None


def test_molecule_smiles_prefers_standardized_structure(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "standardized.csv").write_text(
        "molecule_id,smiles,canonical_smiles,valid_smiles\n"
        "mol_a,OCC,CCO,True\n",
        encoding="utf-8",
    )
    loaded = app.load_output_directory(output_dir)

    assert app.molecule_smiles_from_outputs(loaded, "mol_a") == "CCO"


def test_report_status_message_does_not_expose_path(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "compound_intelligence_report_mol_a.md"

    status, message = app.report_status_message(report_path)

    assert status == "info"
    assert message == "No molecule report has been generated yet."
    assert str(report_path) not in message

    report_path.parent.mkdir()
    report_path.write_text("# Report", encoding="utf-8")
    status, message = app.report_status_message(report_path)

    assert status == "success"
    assert message == "A molecule report is available for this candidate."
    assert str(report_path) not in message


def test_identity_acronym_labels_are_preserved() -> None:
    assert app.display_label("inchikey") == "InChIKey"
    assert app.display_label("pubchem_cid") == "PubChem CID"
    assert app.display_label("chembl_id") == "ChEMBL ID"
    assert app.display_label("surechembl_id") == "SureChEMBL ID"


def test_safe_run_name_sanitizes_and_uses_timestamp() -> None:
    assert app.safe_run_name("My Run: 01!") == "My_Run_01"
    assert (
        app.safe_run_name("", timestamp=datetime(2026, 6, 17, 12, 30, 5))
        == "run_20260617_123005"
    )


def test_generated_input_validation_accepts_required_columns() -> None:
    df = pd.DataFrame({"molecule_id": ["mol_a"], "smiles": ["CCO"]})

    app.validate_generated_smiles(df)


def test_generated_input_validation_rejects_missing_smiles() -> None:
    df = pd.DataFrame({"molecule_id": ["mol_a"]})

    with pytest.raises(ValueError, match="smiles"):
        app.validate_generated_smiles(df)


def test_uploaded_files_are_saved_to_app_runs(tmp_path: Path) -> None:
    generated = BytesIO(b"molecule_id,smiles\nmol_a,CCO\n")
    reference = BytesIO(b"reference_name,smiles,reference_role,target,notes\nethanol,CCO,control,demo,note\n")

    paths = app.prepare_app_run_inputs(
        run_name="User Upload",
        generated_upload=generated,
        reference_upload=reference,
        text_upload=None,
        app_runs_dir=tmp_path / "app_runs",
    )

    assert paths.run_dir == tmp_path / "app_runs" / "User_Upload"
    assert paths.generated_smiles.exists()
    assert paths.references.exists()
    assert paths.text_evidence.exists()
    assert paths.output_dir.exists()
    references = pd.read_csv(paths.references)
    assert references.columns.tolist() == list(app.REFERENCE_OUTPUT_COLUMNS)
    assert references.loc[0, "reference_name"] == "ethanol"


def test_online_options_are_on_by_default() -> None:
    defaults = app.default_workflow_options()

    assert defaults["online_lookup"] is True
    assert defaults["online_surechembl"] is True
    assert defaults["use_chemberta"] is True
    assert defaults["generate_reports"] is True
    assert defaults["report_only_fully_analyzed"] is True
    assert defaults["max_molecules"] == 10
    assert defaults["report_top_n"] == 5


def test_usage_guide_and_mode_notes_exist() -> None:
    guide = app.usage_guide_markdown()

    assert "### How to use this app" in guide
    for step in app.APP_USAGE_STEPS:
        assert step in guide
    assert app.ONLINE_LOOKUP_NOTE == (
        "Online lookup is enabled by default for new analyses. PubChem, "
        "ChEMBL, and SureChEMBL are called only after you click Run analysis."
    )
    assert app.LOAD_EXISTING_NOTE == (
        "This mode only loads an existing output folder. It does not rerun the "
        "pipeline or call online services."
    )


def test_load_existing_results_does_not_run_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "prioritization_results.csv").write_text(
        "molecule_id,nlp_status\nmol_a,not_run\n",
        encoding="utf-8",
    )

    def unexpected_pipeline_call(*args, **kwargs):
        raise AssertionError("Loading results must not run the pipeline.")

    monkeypatch.setattr(app, "run_pipeline", unexpected_pipeline_call)

    loaded = app.load_output_directory(output_dir)

    assert loaded.tables["prioritization"]["molecule_id"].tolist() == ["mol_a"]
