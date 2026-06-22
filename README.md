# AI Molecule Design Assistant

Step-by-step support for evaluating generated SMILES using chemical identity,
public database evidence, RDKit drug-likeness, ChemBERTa chemical-space
embeddings, text evidence, and final design prioritization.

This Python and Streamlit drug-design helper validates structures, calculates
RDKit descriptors, measures structural similarity, builds conservative
biomedical context, scores local text evidence, and generates molecule-level
reports.

The default workflow is local and reproducible. Online lookups are optional and
are not required for the included demonstration.

## Key Features

- SMILES validation, canonicalization, and molecular descriptors with RDKit
- Exact public chemical-name identification with PubChem, ChEMBL evidence, and
  an optional systematic-name fallback
- Morgan fingerprint and Tanimoto similarity against a public reference panel
- Conservative public biomedical context with similarity-gated annotations
- Local molecule-to-text evidence matching
- Optional ChemBERTa embeddings and UMAP/PCA visualization
- Evidence completeness and computed-analysis status tracking
- Streamlit dashboard for filtering, visualization, and report generation
- Markdown reports with 2D molecular structures
- Automated tests covering pipeline, context, NLP, reporting, and app behavior

## Repository Layout

```text
.
├── app.py
├── data/
│   └── examples/
├── src/
├── tests/
├── environment.yml
├── requirements.txt
└── README.md
```

Generated `outputs/` and `app_runs/` folders are intentionally excluded from
Git. This keeps the repository small and prevents local uploads or generated
artifacts from being committed accidentally.

## Quick Start

The recommended setup uses Python 3.11 in Conda:

```powershell
conda env create -f environment.yml
conda activate molecule-intelligence
```

Run the test suite:

```powershell
python -m pytest -q
```

## Run the Public Demo Pipeline

```powershell
python -m src.pipeline --input data/examples/druglike_candidate_demo.csv --references data/examples/druglike_reference_panel.csv --text-evidence data/examples/text_evidence_demo.csv --output-dir outputs/public_druglike_demo_context_nlp_fixed --use-chemberta --report-top-n 5 --report-dir outputs/public_druglike_demo_context_nlp_fixed/reports --clean-report-dir
```

The example data uses neutral molecule IDs, public reference structures,
synthetic text evidence, and one intentionally invalid SMILES row for validation
testing.

Important generated files include:

- `prioritization_results.csv`
- `compound_context.csv`
- `chemical_identity.csv`
- `text_nlp.csv`
- `similarity_top_hits.csv`
- `chemberta_embeddings.csv`
- `visualization_coordinates.csv`
- `reports/`
- `report_images/`

Because outputs are ignored by Git, rerun the command above after cloning to
regenerate the complete public demonstration.

## Launch the Streamlit App

```powershell
streamlit run app.py
```

The app opens on a clean welcome screen and does not automatically load an old
output folder. Start with **Run public demo** or open **Upload my own SMILES**.
Returning users can optionally load a completed output folder from the sidebar.
No results appear until one of these actions is explicitly submitted.

The public demo is executed incrementally rather than as one hidden batch process.
Starting the tutorial creates a fresh workspace but does not perform any
calculation. At each stage the app first explains:

- what will be calculated;
- why the calculation is useful;
- which input is used; and
- which output will be created.

The user then clicks **Run Step N on public example**. Only that stage runs.
Its table and visualization appear after completion, and the next stage is
unlocked with a continue button.

The tutorial follows these stages:

1. Load and validate SMILES
2. Chemical identity
3. Public database lookup
4. RDKit molecular properties
5. ChemBERTa chemical space
6. Text evidence and biomedical context
7. Final prioritization
8. Reports

Each stage explains its purpose, shows its inputs and generated files, presents
a compact result table, and includes a relevant visualization when available.
The user advances with a clear continue button instead of seeing the entire
dashboard at once.

The command-line pipeline creates artifacts in the same logical order:

1. `standardized.csv`
2. `chemical_identity.csv`
3. `public_lookup.csv` and `surechembl_evidence.csv`
4. `descriptors.csv` plus local reference-similarity outputs
5. `chemberta_embeddings.csv` and `visualization_coordinates.csv` when enabled
6. `compound_context.csv` and `text_nlp.csv`
7. `prioritization_results.csv`
8. Markdown reports

Final prioritization retains evidence status from preceding stages. Unrun
optional stages are labeled `not_run` rather than being represented as a
numeric zero. Reports distinguish evidence that directly contributes numeric
score components from evidence that contributes identity, status, chemical-space,
or biomedical interpretation.

The app includes a short usage guide. In **Run new analysis with my files**, online
PubChem/ChEMBL lookup, SureChEMBL structure evidence search, ChemBERTa
embeddings, top-N reports, and fully analyzed report filtering are selected by
default. No online service is called until the user submits the form by clicking
**Run analysis**.

In **Load existing results**, the app only reads the selected output folder. It
does not rerun the pipeline or call online services.

The dashboard reads local output files and clearly distinguishes available,
not-run, unavailable, and error states. Missing or empty NLP output is explained
for the selected folder. The app does not require online database access for the
public demonstration. UI tables, filters, plot axes, and hover labels use
readable presentation names while generated CSV files retain stable
machine-readable column names.

## Browser Testing

Start the app in the `molecule-intelligence` conda environment:

```powershell
streamlit run app.py
```

Then open `http://localhost:8501` and load a public-safe output folder such as:

```text
app_runs\run_20260617_182940\outputs
```

The automated UI fallback uses Streamlit's app-testing harness. It renders the
page, selects the output folder, checks visible headings, tables, plots, readable
labels, the molecule image, and both report-generation actions without calling
online services:

```powershell
python -m pytest -q tests/test_browser_app.py --basetemp C:\tmp\pytest-molecule-intelligence-browser
```

Run the complete test suite with:

```powershell
python -m pytest -q --basetemp C:\tmp\pytest-molecule-intelligence
```

## Public Biomedical Context

Biological annotations are not transferred solely because a nearest structural
reference exists:

- Similarity at least `0.50`: reference context may be shown with a warning.
- Similarity from `0.30` to below `0.50`: context is marked weak and
  reference-only.
- Similarity below `0.30`: the reference is shown only for structural
  orientation, with no biological target context assigned.

Exact public names and identifiers are populated only when supported by the
available lookup evidence.

`chemical_identity.csv` records the RDKit InChIKey, exact PubChem/ChEMBL
identifiers and names when available, PubChem IUPAC names and synonyms, lookup
status, and confidence. A generated systematic name may be requested from NCI
Cactus only when no PubChem or ChEMBL name is available; it is labeled
`generated_iupac_name_only` and is never treated as an exact public identity.

## Screenshots

Add release screenshots at these suggested paths:

- `docs/screenshots/dashboard-overview.png`
- `docs/screenshots/chemical-space.png`
- `docs/screenshots/compound-report.png`

Example Markdown:

```markdown
![Dashboard overview](docs/screenshots/dashboard-overview.png)
```

## Input Formats

Candidate molecules:

```csv
molecule_id,smiles,notes
demo_001,CCO,Public-safe demonstration row
```

Reference compounds require at least:

```csv
reference_name,smiles
Public reference,CCO
```

Text evidence can use either the molecule-linked schema or the neutral
public-safe schema demonstrated in `data/examples/text_evidence_demo.csv`.

## Reproducibility Notes

- Python version: 3.11
- Dependencies are listed in both `environment.yml` and `requirements.txt`.
- Tests do not call online APIs.
- ChemBERTa must already be available in the local model cache when used
  offline.
- ChemBERTa is unavailable for invalid SMILES.

## Scope

This repository supports research workflow development using public or
synthetic example data. Its outputs are evidence summaries and design-ranking
aids that require expert review.
