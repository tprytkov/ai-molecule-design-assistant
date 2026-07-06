"""Generate ADMET descriptor/rule fallback outputs from molecule SMILES."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from src.admet.admet_schema import (
    ADMET_EVIDENCE_NOTE,
    ADMET_PREDICTION_COLUMNS,
    ADMET_SUMMARY_COLUMNS,
    MODEL_BACKEND_FALLBACK,
    MODEL_CACHE_STATUS_FALLBACK,
    MODEL_ID_FALLBACK,
    MODEL_STATUS_FALLBACK,
    TRAINING_DATASET_FALLBACK,
)
from src.admet.admet_model_selection import (
    VALIDATION_STATUS_BENCHMARK_PASSED,
    VALIDATION_STATUS_EXPERIMENTAL,
    VALIDATION_STATUS_NOT_EVALUATED,
    allow_experimental_admet_model_use,
    validation_status_allows_prediction,
    validation_status_for_model,
)
from src.model_source_status import HUGGINGFACE_CACHE_DIR, model_is_cached
from src.optional_domain_models import CHEMBERTA_BBB_MODEL_ID


ENDPOINT_BBB = "bbb_permeability_cns_likeness"
ENDPOINT_SOLUBILITY = "solubility_esol_style"
ENDPOINT_LOGP = "lipophilicity_logp"
ENDPOINT_HERG = "herg_cardiotoxicity_descriptor_triage"
ENDPOINT_CYP = "cyp_inhibition_descriptor_triage"
ENDPOINT_TOX = "general_toxicity_descriptor_triage"
ENDPOINTS = (
    ENDPOINT_BBB,
    ENDPOINT_SOLUBILITY,
    ENDPOINT_LOGP,
    ENDPOINT_HERG,
    ENDPOINT_CYP,
    ENDPOINT_TOX,
)
MODEL_STATUS_TUNED_BBB = "experimental_public_hf_model"
MODEL_BACKEND_TUNED_BBB = "transformers_sequence_classification"
MODEL_CACHE_STATUS_CACHED = "cached"
TRAINING_DATASET_TUNED_BBB = "public_hf_model_card_bbb_permeability"
TUNED_BBB_EVIDENCE_NOTE = (
    "Experimental public Hugging Face tuned ChemBERTa BBB classifier for "
    "computational research triage only; not experimental ADMET, safety, "
    "toxicity, or clinical evidence."
)


@dataclass(frozen=True)
class ADMETDescriptorRecord:
    """Descriptor values needed by the ADMET fallback rules."""

    molecule_id: str
    smiles: str
    valid: bool
    molecular_weight: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    rotatable_bonds: int | None = None
    aromatic_rings: int | None = None
    heavy_atoms: int | None = None
    aromatic_heavy_atoms: int | None = None
    error_message: str = ""


class TunedBBBClassifierUnavailable(RuntimeError):
    """Raised when the optional tuned BBB classifier cannot be loaded locally."""


class TunedChembertaBBBClassifier:
    """Cached local ChemBERTa sequence classifier for BBB triage."""

    model_id = CHEMBERTA_BBB_MODEL_ID

    def __init__(
        self,
        *,
        model_id: str = CHEMBERTA_BBB_MODEL_ID,
        cache_dir: Path = HUGGINGFACE_CACHE_DIR,
        local_files_only: bool = True,
    ) -> None:
        self.model_id = model_id
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise TunedBBBClassifierUnavailable(
                "transformers and torch are required for the tuned BBB classifier."
            ) from exc
        try:
            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(cache_dir),
                local_files_only=local_files_only,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                cache_dir=str(cache_dir),
                local_files_only=local_files_only,
            )
            self.model.eval()
        except Exception as exc:
            raise TunedBBBClassifierUnavailable(
                f"Model '{model_id}' is not available in the app-managed cache."
            ) from exc

    def predict_label(self, smiles: str) -> tuple[str, str]:
        """Return conservative label/probability strings for one SMILES."""
        encoded = self.tokenizer(
            [smiles],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = self._torch.nn.functional.softmax(logits, dim=-1)[0]
        index = int(probabilities.argmax().item())
        probability = float(probabilities[index].item())
        raw_label = str(getattr(self.model.config, "id2label", {}).get(index, index)).lower()
        if any(term in raw_label for term in ("non", "imperme", "negative", "false", "0")):
            label = "caution"
        elif any(term in raw_label for term in ("perme", "positive", "true", "1")):
            label = "favorable"
        else:
            label = "moderate"
        return label, f"{probability:.3f}"


def load_tuned_bbb_classifier() -> TunedChembertaBBBClassifier | None:
    """Load the tuned BBB classifier only if it appears cached locally."""
    if not model_is_cached(CHEMBERTA_BBB_MODEL_ID):
        return None
    try:
        return TunedChembertaBBBClassifier()
    except TunedBBBClassifierUnavailable:
        return None


def parse_bool(value: object) -> bool:
    """Parse common CSV boolean representations."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: object) -> float | None:
    """Parse a float if present."""
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: object) -> int | None:
    """Parse an integer if present."""
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else None


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    """Read CSV rows from an optional path."""
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def descriptor_from_smiles(
    molecule_id: str,
    smiles: str,
    *,
    upstream_valid: bool = True,
    upstream_error: str = "",
) -> ADMETDescriptorRecord:
    """Calculate required ADMET descriptors directly from SMILES."""
    cleaned = str(smiles or "").strip()
    if not upstream_valid:
        return ADMETDescriptorRecord(
            molecule_id=molecule_id,
            smiles=cleaned,
            valid=False,
            error_message=upstream_error or "Input row is marked invalid.",
        )
    if not cleaned:
        return ADMETDescriptorRecord(
            molecule_id=molecule_id,
            smiles=cleaned,
            valid=False,
            error_message="SMILES is missing.",
        )
    molecule = Chem.MolFromSmiles(cleaned)
    if molecule is None:
        return ADMETDescriptorRecord(
            molecule_id=molecule_id,
            smiles=cleaned,
            valid=False,
            error_message="RDKit could not parse SMILES.",
        )
    aromatic_heavy_atoms = sum(
        1 for atom in molecule.GetAtoms() if atom.GetIsAromatic() and atom.GetAtomicNum() > 1
    )
    return ADMETDescriptorRecord(
        molecule_id=molecule_id,
        smiles=Chem.MolToSmiles(molecule),
        valid=True,
        molecular_weight=Descriptors.MolWt(molecule),
        logp=Crippen.MolLogP(molecule),
        tpsa=rdMolDescriptors.CalcTPSA(molecule),
        hbd=rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        hba=rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        rotatable_bonds=Lipinski.NumRotatableBonds(molecule),
        aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(molecule),
        heavy_atoms=molecule.GetNumHeavyAtoms(),
        aromatic_heavy_atoms=aromatic_heavy_atoms,
    )


def descriptor_from_row(row: Mapping[str, str]) -> ADMETDescriptorRecord:
    """Build an ADMET descriptor record from descriptor or standardized rows."""
    molecule_id = str(row.get("molecule_id", "")).strip()
    smiles = str(
        row.get("canonical_smiles")
        or row.get("smiles")
        or row.get("SMILES")
        or ""
    ).strip()
    upstream_valid = parse_bool(row.get("valid_smiles", "true"))
    base = descriptor_from_smiles(
        molecule_id,
        smiles,
        upstream_valid=upstream_valid,
        upstream_error=str(row.get("descriptor_error") or row.get("parse_error") or ""),
    )
    if not base.valid:
        return base
    return ADMETDescriptorRecord(
        molecule_id=molecule_id,
        smiles=base.smiles,
        valid=True,
        molecular_weight=parse_float(row.get("molecular_weight")) or base.molecular_weight,
        logp=parse_float(row.get("logp")) or base.logp,
        tpsa=parse_float(row.get("tpsa")) or base.tpsa,
        hbd=parse_int(row.get("hbd")) if parse_int(row.get("hbd")) is not None else base.hbd,
        hba=parse_int(row.get("hba")) if parse_int(row.get("hba")) is not None else base.hba,
        rotatable_bonds=(
            parse_int(row.get("rotatable_bonds"))
            if parse_int(row.get("rotatable_bonds")) is not None
            else base.rotatable_bonds
        ),
        aromatic_rings=(
            parse_int(row.get("aromatic_rings"))
            if parse_int(row.get("aromatic_rings")) is not None
            else base.aromatic_rings
        ),
        heavy_atoms=base.heavy_atoms,
        aromatic_heavy_atoms=base.aromatic_heavy_atoms,
    )


def descriptor_records(
    *,
    descriptors_path: Path | None = None,
    standardized_path: Path | None = None,
) -> list[ADMETDescriptorRecord]:
    """Load descriptor rows, preferring descriptors.csv when available."""
    rows = read_csv_rows(descriptors_path)
    if not rows:
        rows = read_csv_rows(standardized_path)
    return [descriptor_from_row(row) for row in rows]


def label_bbb(record: ADMETDescriptorRecord) -> tuple[str, str]:
    """Return BBB/CNS-likeness label and value text."""
    if not record.valid or None in (record.tpsa, record.logp, record.hbd, record.molecular_weight):
        return "unavailable", "unavailable"
    assert record.tpsa is not None
    assert record.logp is not None
    assert record.hbd is not None
    assert record.molecular_weight is not None
    value = f"TPSA={record.tpsa:.1f}; LogP={record.logp:.2f}; HBD={record.hbd}; MW={record.molecular_weight:.1f}"
    if record.tpsa <= 90 and 1 <= record.logp <= 4 and record.hbd <= 2 and record.molecular_weight <= 450:
        return "favorable", value
    if record.tpsa <= 120 and -1 <= record.logp <= 5 and record.hbd <= 3 and record.molecular_weight <= 550:
        return "moderate", value
    return "caution", value


def esol_style_logs(record: ADMETDescriptorRecord) -> float | None:
    """Return a conservative ESOL-style logS estimate."""
    if not record.valid or None in (
        record.logp,
        record.molecular_weight,
        record.rotatable_bonds,
        record.heavy_atoms,
        record.aromatic_heavy_atoms,
    ):
        return None
    assert record.logp is not None
    assert record.molecular_weight is not None
    assert record.rotatable_bonds is not None
    assert record.heavy_atoms is not None
    assert record.aromatic_heavy_atoms is not None
    aromatic_proportion = (
        record.aromatic_heavy_atoms / record.heavy_atoms if record.heavy_atoms else 0.0
    )
    return (
        0.16
        - 0.63 * record.logp
        - 0.0062 * record.molecular_weight
        + 0.066 * record.rotatable_bonds
        - 0.74 * aromatic_proportion
    )


def label_solubility(record: ADMETDescriptorRecord) -> tuple[str, str]:
    """Return solubility label and ESOL-style value."""
    logs = esol_style_logs(record)
    if logs is None:
        return "unavailable", "unavailable"
    if logs >= -4:
        return "favorable", f"logS={logs:.2f}"
    if logs >= -6:
        return "moderate", f"logS={logs:.2f}"
    return "caution", f"logS={logs:.2f}"


def label_logp(record: ADMETDescriptorRecord) -> tuple[str, str]:
    """Return lipophilicity label and LogP value."""
    if not record.valid or record.logp is None:
        return "unavailable", "unavailable"
    if -1 <= record.logp <= 3:
        return "favorable", f"LogP={record.logp:.2f}"
    if -2 <= record.logp <= 5:
        return "moderate", f"LogP={record.logp:.2f}"
    return "caution", f"LogP={record.logp:.2f}"


def label_herg(record: ADMETDescriptorRecord) -> tuple[str, str]:
    """Return hERG/cardiotoxicity descriptor-triage label."""
    if not record.valid or None in (record.logp, record.molecular_weight, record.aromatic_rings):
        return "unavailable", "unavailable"
    assert record.logp is not None
    assert record.molecular_weight is not None
    assert record.aromatic_rings is not None
    value = f"LogP={record.logp:.2f}; MW={record.molecular_weight:.1f}; aromatic_rings={record.aromatic_rings}"
    if record.logp > 4.0 and (record.molecular_weight > 400 or record.aromatic_rings >= 3):
        return "caution", value
    if record.logp > 3.5 or record.molecular_weight > 450 or record.aromatic_rings >= 3:
        return "moderate", value
    return "favorable", value


def label_cyp(record: ADMETDescriptorRecord) -> tuple[str, str]:
    """Return CYP inhibition descriptor-triage label."""
    if not record.valid or None in (record.logp, record.molecular_weight, record.aromatic_rings):
        return "unavailable", "unavailable"
    assert record.logp is not None
    assert record.molecular_weight is not None
    assert record.aromatic_rings is not None
    value = f"LogP={record.logp:.2f}; MW={record.molecular_weight:.1f}; aromatic_rings={record.aromatic_rings}"
    if record.logp >= 5 or record.molecular_weight >= 500 or record.aromatic_rings >= 4:
        return "caution", value
    if record.logp >= 3.5 or record.molecular_weight >= 425 or record.aromatic_rings >= 3:
        return "moderate", value
    return "favorable", value


def label_toxicity(record: ADMETDescriptorRecord) -> tuple[str, str]:
    """Return general toxicity descriptor-triage label."""
    if not record.valid or None in (record.logp, record.tpsa, record.molecular_weight):
        return "unavailable", record.error_message or "unavailable"
    assert record.logp is not None
    assert record.tpsa is not None
    assert record.molecular_weight is not None
    caution_count = sum(
        (
            record.logp > 5,
            record.tpsa > 140,
            record.molecular_weight > 600,
            (record.hbd or 0) > 5,
            (record.hba or 0) > 10,
        )
    )
    value = f"descriptor_caution_flags={caution_count}"
    if caution_count >= 2:
        return "caution", value
    if caution_count == 1:
        return "moderate", value
    return "favorable", value


def endpoint_label(record: ADMETDescriptorRecord, endpoint: str) -> tuple[str, str]:
    """Return label and value for one endpoint."""
    if endpoint == ENDPOINT_BBB:
        return label_bbb(record)
    if endpoint == ENDPOINT_SOLUBILITY:
        return label_solubility(record)
    if endpoint == ENDPOINT_LOGP:
        return label_logp(record)
    if endpoint == ENDPOINT_HERG:
        return label_herg(record)
    if endpoint == ENDPOINT_CYP:
        return label_cyp(record)
    return label_toxicity(record)


def prediction_row(
    record: ADMETDescriptorRecord,
    endpoint: str,
    *,
    bbb_classifier: TunedChembertaBBBClassifier | None = None,
    bbb_validation_status: str = VALIDATION_STATUS_NOT_EVALUATED,
) -> dict[str, str]:
    """Build one admet_predictions.csv row."""
    label, value = endpoint_label(record, endpoint)
    probability = ""
    model_id = MODEL_ID_FALLBACK
    model_backend = MODEL_BACKEND_FALLBACK
    model_status = MODEL_STATUS_FALLBACK
    model_cache_status = MODEL_CACHE_STATUS_FALLBACK
    validation_status = VALIDATION_STATUS_NOT_EVALUATED
    fallback_used = "yes"
    training_dataset = TRAINING_DATASET_FALLBACK
    evidence_note = ADMET_EVIDENCE_NOTE
    if endpoint == ENDPOINT_BBB and bbb_classifier is not None and record.valid:
        try:
            label, probability = bbb_classifier.predict_label(record.smiles)
            value = f"tuned_chemberta_bbb_probability={probability}"
            model_id = bbb_classifier.model_id
            model_backend = MODEL_BACKEND_TUNED_BBB
            model_status = MODEL_STATUS_TUNED_BBB
            model_cache_status = MODEL_CACHE_STATUS_CACHED
            validation_status = bbb_validation_status
            fallback_used = "no"
            training_dataset = TRAINING_DATASET_TUNED_BBB
            evidence_note = TUNED_BBB_EVIDENCE_NOTE
        except Exception:
            label, value = endpoint_label(record, endpoint)
    return {
        "molecule_id": record.molecule_id,
        "smiles": record.smiles,
        "admet_endpoint": endpoint,
        "prediction_value": value,
        "prediction_probability": probability,
        "prediction_label": label,
        "model_id": model_id,
        "model_backend": model_backend,
        "model_status": model_status,
        "model_cache_status": model_cache_status,
        "validation_status": validation_status,
        "fallback_used": fallback_used,
        "training_dataset": training_dataset,
        "evidence_note": evidence_note,
    }


def readiness_category(labels: Iterable[str]) -> str:
    """Summarize endpoint labels into one ADMET readiness category."""
    values = list(labels)
    if not values or all(value == "unavailable" for value in values):
        return "unavailable"
    if "caution" in values:
        return "caution"
    if "moderate" in values:
        return "moderate"
    return "favorable"


def summary_row(
    record: ADMETDescriptorRecord,
    predictions: Sequence[dict[str, str]],
) -> dict[str, str]:
    """Build one admet_summary.csv row."""
    by_endpoint = {
        row["admet_endpoint"]: row["prediction_label"] for row in predictions
    }
    toxicity_labels = (
        by_endpoint.get(ENDPOINT_HERG, "unavailable"),
        by_endpoint.get(ENDPOINT_CYP, "unavailable"),
        by_endpoint.get(ENDPOINT_TOX, "unavailable"),
    )
    return {
        "molecule_id": record.molecule_id,
        "smiles": record.smiles,
        "bbb_prediction_label": by_endpoint.get(ENDPOINT_BBB, "unavailable"),
        "cns_property_flag": by_endpoint.get(ENDPOINT_BBB, "unavailable"),
        "toxicity_risk_flag": readiness_category(toxicity_labels),
        "admet_readiness_category": readiness_category(by_endpoint.values()),
        "model_status": (
            "mixed_tuned_bbb_and_descriptor_rule"
            if any(row.get("model_status") == MODEL_STATUS_TUNED_BBB for row in predictions)
            else MODEL_STATUS_FALLBACK
        ),
        "validation_status": (
            VALIDATION_STATUS_BENCHMARK_PASSED
            if any(
                row.get("validation_status") == VALIDATION_STATUS_BENCHMARK_PASSED
                for row in predictions
            )
            else (
                VALIDATION_STATUS_EXPERIMENTAL
                if any(
                    row.get("validation_status") == VALIDATION_STATUS_EXPERIMENTAL
                    for row in predictions
                )
                else VALIDATION_STATUS_NOT_EVALUATED
            )
        ),
        "fallback_used": (
            "yes" if any(row.get("fallback_used", "yes") == "yes" for row in predictions) else "no"
        ),
        "evidence_note": ADMET_EVIDENCE_NOTE,
    }


def build_admet_outputs(
    records: Iterable[ADMETDescriptorRecord],
    *,
    bbb_classifier: TunedChembertaBBBClassifier | None = None,
    bbb_validation_status: str = VALIDATION_STATUS_BENCHMARK_PASSED,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build ADMET prediction and summary rows."""
    prediction_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for record in records:
        molecule_predictions = [
            prediction_row(
                record,
                endpoint,
                bbb_classifier=bbb_classifier,
                bbb_validation_status=bbb_validation_status,
            )
            for endpoint in ENDPOINTS
        ]
        prediction_rows.extend(molecule_predictions)
        summary_rows.append(summary_row(record, molecule_predictions))
    return prediction_rows, summary_rows


def write_csv(path: Path, rows: Iterable[Mapping[str, str]], columns: Sequence[str]) -> int:
    """Write rows using a fixed schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def admet_csv(
    *,
    descriptors_path: Path | None = None,
    standardized_path: Path | None = None,
    predictions_path: Path,
    summary_path: Path,
    use_tuned_bbb_if_cached: bool = True,
    admet_evaluation_summary_path: Path | None = None,
    allow_experimental_tuned_bbb: bool | None = None,
) -> dict[str, int]:
    """Generate ADMET descriptor/rule fallback CSV outputs."""
    records = descriptor_records(
        descriptors_path=descriptors_path,
        standardized_path=standardized_path,
    )
    bbb_validation_status = validation_status_for_model(
        model_id=CHEMBERTA_BBB_MODEL_ID,
        endpoint_name="bbb_permeability",
        summary_path=admet_evaluation_summary_path,
    )
    experimental_allowed = (
        allow_experimental_admet_model_use()
        if allow_experimental_tuned_bbb is None
        else allow_experimental_tuned_bbb
    )
    if experimental_allowed and bbb_validation_status == VALIDATION_STATUS_NOT_EVALUATED:
        bbb_validation_status = VALIDATION_STATUS_EXPERIMENTAL
    bbb_allowed = (
        use_tuned_bbb_if_cached
        and validation_status_allows_prediction(
            bbb_validation_status,
            allow_experimental=experimental_allowed,
        )
    )
    bbb_classifier = load_tuned_bbb_classifier() if bbb_allowed else None
    predictions, summaries = build_admet_outputs(
        records,
        bbb_classifier=bbb_classifier,
        bbb_validation_status=bbb_validation_status,
    )
    return {
        "predictions": write_csv(
            predictions_path,
            predictions,
            ADMET_PREDICTION_COLUMNS,
        ),
        "summary": write_csv(summary_path, summaries, ADMET_SUMMARY_COLUMNS),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the ADMET fallback CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate ADMET descriptor/rule fallback outputs."
    )
    parser.add_argument("--descriptors", type=Path)
    parser.add_argument("--standardized", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ADMET fallback generation from the command line."""
    args = build_parser().parse_args(argv)
    counts = admet_csv(
        descriptors_path=args.descriptors,
        standardized_path=args.standardized,
        predictions_path=args.predictions,
        summary_path=args.summary,
    )
    print(
        "Wrote "
        f"{counts['predictions']} ADMET prediction rows and "
        f"{counts['summary']} ADMET summary rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
