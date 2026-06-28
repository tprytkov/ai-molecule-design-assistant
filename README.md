# AI Molecule Design Assistant

## Project Motivation

Generative molecular models can rapidly produce large numbers of chemically valid and drug-like candidate structures. However, generated molecules are not automatically useful for discovery or intellectual-property planning. Many generated structures may be invalid, already known, too close to public compounds, poorly drug-like, weakly supported by evidence, or difficult to interpret.

This project was developed as a practical research assistant for exploring molecules produced by generative AI workflows. The goal is to help researchers move from a raw list of generated SMILES to an interpretable, evidence-supported set of prioritized candidates.

The application evaluates generated molecules through several complementary questions:

1. Is the generated SMILES chemically valid and standardized?
2. Does the molecule have an exact public identity in databases such as PubChem or ChEMBL?
3. Is the molecule close to known reference compounds or structurally differentiated from them?
4. Does the molecule have acceptable RDKit-based drug-likeness properties?
5. Where does the molecule sit in ChemBERTa chemical-embedding space?
6. Is there relevant text evidence or biological context supporting further review?
7. Does public structure evidence suggest that the molecule is already represented in public chemical or patent-derived resources?
8. Which generated molecules should be prioritized for additional computational, experimental, or expert review?

In this context, “IP potential” is used as an early-stage research signal. It reflects whether a generated molecule appears chemically differentiated from exact public matches or close public analogs while retaining favorable drug-like properties and interpretable evidence. This score is intended to support triage and hypothesis generation. It is not a legal opinion and does not determine patentability, novelty, freedom to operate, ownership, infringement risk, efficacy, safety, or clinical value.

The intended use case is therefore not to replace expert review, but to organize generated molecular candidates into a more interpretable decision space before investing time in synthesis, docking, biological testing, patent review, or medicinal chemistry optimization.

## About the workflow

### 1. SMILES validation and standardization

Generated SMILES are parsed and standardized before downstream analysis. This
establishes a chemically interpretable representation for identifier
generation, descriptor calculation, fingerprints, and molecular embeddings.
[RDKit documentation](https://www.rdkit.org/docs/) describes the open-source
cheminformatics functionality used for molecular parsing and representation.

### 2. Chemical identity lookup

Standardized structures are assigned structure-derived identifiers such as
InChIKey and are checked for exact public records when lookup services are
enabled. [PubChem PUG-REST](https://pubmed.ncbi.nlm.nih.gov/27424744/) provides
programmatic access to PubChem identifiers and properties, while
[ChEMBL web services](https://academic.oup.com/nar/article/43/W1/W612/2467881)
provide access to curated compound and bioactivity records.

### 3. Public database evidence

Exact PubChem or ChEMBL matches indicate that a standardized structure is
represented in those public resources.
[SureChEMBL](https://academic.oup.com/nar/article/44/D1/D1220/2503102) can add
structure-level evidence extracted from chemically annotated patent documents.
These results are interpreted only as public structure evidence.

### 4. RDKit drug-likeness

RDKit descriptors, fingerprint similarity, Lipinski-style property checks, and
QED are presented as design heuristics. The
[Lipinski framework](https://doi.org/10.1016/S0169-409X(00)00129-0) summarizes
empirical property ranges, and [QED](https://doi.org/10.1038/nchem.1243)
combines molecular-property distributions into a quantitative drug-likeness
estimate. Neither constitutes evidence of biological activity or safety.

### 5. ChemBERTa chemical-space embeddings

[ChemBERTa](https://arxiv.org/abs/2010.09885) uses transformer-based
self-supervised learning on SMILES to construct molecular representations.
Low-dimensional views may use [UMAP](https://arxiv.org/abs/1802.03426) to
display clusters, outliers, and reference-like neighborhoods. Generated and
reference molecules are fit into the same projection when reference molecules
are available, with nearest-reference fingerprint similarity shown for visual
triage. The dashboard colors and shapes generated versus reference molecules
separately, and can optionally show nearest-reference links and a compact
similarity distribution. These plots are exploratory representations rather
than experimental validation.

### 6. Biomedical evidence and biological context

After molecular identity and context are available, an optional configurable
sentence-transformer compatible biomedical encoder compares molecule-context
summaries with user-provided biomedical evidence text. Local users may point
this step at a cached BioBERT/PubMedBERT-style sentence embedding model, while
the public Streamlit app writes a valid skipped output if the configured model
is unavailable. The [Sentence-BERT](https://arxiv.org/abs/1908.10084) approach
and the [Sentence Transformers documentation](https://www.sbert.net/) describe
the methodological and software basis for efficient semantic similarity
matching. This stage organizes evidence for review and hypothesis generation;
it does not establish biological activity.

### 7. Patent/IP-context evidence

Patent/IP-context evidence is represented as a separate optional evidence
embedding stage. Local users may point this step at a cached PaECTER,
patent-BERT-style, or other sentence-transformer compatible patent encoder.
The public Streamlit app does not require that model: if it is unavailable, the
workflow writes a schema-valid skipped output while preserving SureChEMBL
structure evidence and patent document metadata separately. These outputs are
early research triage signals and do not determine patentability, novelty,
freedom to operate, ownership, or infringement risk.

### 8. Final design prioritization

The final ranking integrates chemical identity, public-database status, RDKit
drug-likeness, reference similarity, ChemBERTa chemical-space context,
text-evidence matching, and evidence completeness. It is a transparent
research-prioritization aid for selecting candidates for further computational
or experimental review, not a prediction of efficacy, safety, novelty, or
clinical value.

The default workflow is local and reproducible. Online lookups are optional and
are not required for the included guided example.

## Optional embedding models

Step 6 and Step 7 are cloud-safe optional embedding stages. The app can write
valid skipped outputs when an embedding model is unavailable, and skipped
biomedical or patent-text evidence is not treated as a workflow error.

For local runs, these stages use sentence-transformer compatible models only
when the configured model is already available in the local cache. The app does
not automatically download large embedding models on Streamlit Cloud.

Recommended biomedical model category: BioBERT/PubMedBERT-style sentence
embedding models for biomedical evidence and biological-context matching.

Recommended patent model category: PaECTER/patent-BERT-style embedding models
for patent/IP-context text matching.

The default biomedical configuration remains a lightweight general
sentence-transformer baseline so the workflow can run safely in constrained
environments. Users who need domain-specific biomedical or patent semantics can
configure cached local models without making those models required for app
startup or public deployment.

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
├── environment-local.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Generated `outputs/` and `app_runs/` folders are intentionally excluded from
Git. This keeps the repository small and prevents local uploads or generated
artifacts from being committed accidentally.

## Quick Start

Streamlit Cloud installs the deployed app from the root-level
`requirements.txt` file. The root-level `packages.txt` file lists Linux system
libraries used by RDKit structure drawing on Streamlit Cloud. For local
development, the recommended setup uses Python 3.11 in Conda:

```powershell
conda env create -f environment-local.yml
conda activate molecule-intelligence
```

The optional `requirements-dev.txt` file lists pytest and Playwright browser
testing dependencies for non-Conda development environments.

Run the test suite:

```powershell
python -m pytest -q
```

## Run the Guided Example Pipeline

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
- `biomedical_evidence.csv`
- `patent_evidence_embeddings.csv`
- `similarity_top_hits.csv`
- `chemberta_embeddings.csv`
- `visualization_coordinates.csv`
- `reports/`
- `report_images/`

Because outputs are ignored by Git, rerun the command above after cloning to
regenerate the complete public example.

## Launch the Streamlit App

```powershell
streamlit run app.py
```

The app opens on a clean welcome screen and does not automatically load an old
output folder. Start with **Guided example workflow** to learn each evaluation
stage, or open **Upload my own SMILES**. Returning users can optionally load a
completed output folder from the sidebar. No results appear until one of these
actions is explicitly submitted.

The guided example is executed incrementally rather than as one hidden batch
process. Starting it creates a fresh workspace but does not perform any
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
6. Biomedical evidence and biological context
7. Patent/IP-context evidence
8. Final prioritization
9. Reports

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
6. `compound_context.csv`, legacy-compatible `text_nlp.csv`, and molecule-level
   `biomedical_evidence.csv`
7. `patent_evidence_embeddings.csv`
8. `prioritization_results.csv`
9. Markdown reports

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
guided example. UI tables, filters, plot axes, and hover labels use
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

To smoke-test the deployed Streamlit app without running online lookup steps or
model downloads, provide the public app URL as an environment variable:

```powershell
$env:STREAMLIT_APP_URL = "https://your-streamlit-app-url"
python -m pytest -q tests/test_deployed_app_browser.py --basetemp C:\tmp\pytest-molecule-intelligence-browser
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
- Streamlit Cloud deployment dependencies are listed in `requirements.txt`.
- Local Conda dependencies are listed in `environment-local.yml`.
- Non-Conda test tooling is listed in `requirements-dev.txt`.
- Tests do not call online APIs.
- ChemBERTa must already be available in the local model cache when used
  offline.
- ChemBERTa is unavailable for invalid SMILES.

## Scope

This repository supports research workflow development using public or
synthetic example data. Its outputs are evidence summaries and design-ranking
aids that require expert review.
