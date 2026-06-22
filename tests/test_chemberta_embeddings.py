import csv
import json
from pathlib import Path

from src.chemberta_embeddings import (
    EMBEDDING_COLUMNS,
    VISUALIZATION_COLUMNS,
    chemberta_embeddings_csv,
    merge_chemberta_into_prioritized,
    reduce_embeddings,
    visualization_coordinates_csv,
)


class MockChembertaEmbedder:
    model_name = "mock-chemberta"

    def embed(self, smiles: str) -> list[float]:
        base = float(len(smiles))
        return [base, base / 2.0, 1.0]


def write_standardized(path: Path) -> None:
    path.write_text(
        "molecule_id,canonical_smiles,valid_smiles,error_message\n"
        "mol_a,CCO,True,\n"
        "invalid,,False,Invalid molecule.\n",
        encoding="utf-8",
    )


def write_prioritized(path: Path) -> None:
    path.write_text(
        "molecule_id,prioritization_score_with_nlp,novelty_flag,"
        "ip_potential_category,known_public_match,best_reference_name,"
        "tanimoto_similarity\n"
        "mol_a,0.750,chemically_differentiated_candidate,"
        "higher_ip_potential_signal,False,ethanol,0.900\n"
        "invalid,0.000,not_available,not_available,False,,\n",
        encoding="utf-8",
    )


def test_chemberta_embeddings_retains_invalid_rows(tmp_path: Path) -> None:
    standardized = tmp_path / "standardized.csv"
    output = tmp_path / "embeddings.csv"
    write_standardized(standardized)

    count = chemberta_embeddings_csv(
        standardized,
        output,
        embedder=MockChembertaEmbedder(),
    )

    assert count == 2
    with output.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == EMBEDDING_COLUMNS
        rows = list(reader)
    assert rows[0]["embedding_available"] == "True"
    assert rows[0]["embedding_dim"] == "3"
    assert json.loads(rows[0]["embedding_json"])
    assert rows[1]["embedding_available"] == "False"
    assert rows[1]["error_message"] == "Invalid molecule."


def test_visualization_coordinates_and_prioritized_merge(tmp_path: Path) -> None:
    standardized = tmp_path / "standardized.csv"
    embeddings = tmp_path / "embeddings.csv"
    prioritized = tmp_path / "prioritized.csv"
    coordinates = tmp_path / "coordinates.csv"
    write_standardized(standardized)
    write_prioritized(prioritized)
    chemberta_embeddings_csv(
        standardized,
        embeddings,
        embedder=MockChembertaEmbedder(),
    )

    merge_chemberta_into_prioritized(prioritized, embeddings)
    count = visualization_coordinates_csv(embeddings, prioritized, coordinates)

    assert count == 2
    with prioritized.open("r", encoding="utf-8", newline="") as prioritized_file:
        prioritized_rows = list(csv.DictReader(prioritized_file))
    assert prioritized_rows[0]["chemberta_embedding_available"] == "True"
    assert prioritized_rows[0]["chemberta_embedding_dim"] == "3"

    with coordinates.open("r", encoding="utf-8", newline="") as coord_file:
        reader = csv.DictReader(coord_file)
        assert tuple(reader.fieldnames or ()) == VISUALIZATION_COLUMNS
        coord_rows = list(reader)
    assert coord_rows[0]["molecule_id"] == "mol_a"
    assert coord_rows[0]["coordinate_method"] == "pca"
    assert coord_rows[1]["coordinate_method"] == "not_available"


def test_larger_embedding_sets_use_pca_fallback() -> None:
    vectors = [[float(index), float(index % 3), 1.0] for index in range(31)]

    method, coordinates = reduce_embeddings(vectors)

    assert method == "pca"
    assert len(coordinates) == 31
