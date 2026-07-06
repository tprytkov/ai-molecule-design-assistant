# Target-Specific Structural Demo

This package is separate from the default public molecule-processing demo. It is
a small ADORA2A/xanthine example for exercising target-aware structural
prioritization in the Streamlit app.

Files:

- `target_profile.csv`: human adenosine A2A receptor metadata using public
  ADORA2A / PDB 3RFM context.
- `reference_ligands.csv`: caffeine, theophylline, and xanthine reference
  ligands for structural comparison.
- `demo_molecules.csv`: matched xanthine-like molecules for this target package.
- `demo_docking_results.csv`: illustrative docking-style rows with matching
  `target_id` values for workflow validation.

The docking scores are demo/computational triage values, not experimentally
measured binding affinity. This package does not claim biological activity,
efficacy, safety, or clinical evidence. Replace the target profile and docking
CSV with project-specific inputs before making scientific decisions.
