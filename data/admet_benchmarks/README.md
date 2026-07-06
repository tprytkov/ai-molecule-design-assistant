# ADMET Benchmark Inputs

This folder is reserved for local, public-safe ADMET benchmark CSV files used by
`scripts/evaluate_admet_models.py`.

Expected local CSV schema:

- `molecule_id`
- `smiles`
- `label`
- `split`

Use public benchmark sources such as TDC ADMET or MoleculeNet-style exports when
their licenses allow local evaluation. Benchmark CSV/TSV files are ignored by
Git by default so large downloaded datasets are not committed.

The app does not use SwissADME and does not scrape SwissADME. Benchmark results
are computational model-validation summaries only; they are not experimental
ADMET, toxicity, safety, or clinical evidence.
