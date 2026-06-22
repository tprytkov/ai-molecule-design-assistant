"""Real-browser smoke tests for the local Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_URL = "http://localhost:8501"
PRIMARY_OUTPUT = PROJECT_ROOT / "app_runs" / "run_20260617_182940" / "outputs"
FALLBACK_OUTPUT = PROJECT_ROOT / "outputs" / "public_druglike_demo_context_nlp_fixed"


def app_is_ready() -> bool:
    """Return whether Streamlit is accepting local HTTP requests."""
    try:
        with urllib.request.urlopen(APP_URL, timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def streamlit_server() -> Iterator[None]:
    """Connect to an existing app or start a temporary local Streamlit server."""
    process: subprocess.Popen[str] | None = None
    if not app_is_ready():
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless=true",
                "--server.port=8501",
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for _ in range(60):
            if app_is_ready():
                break
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming ready.")
            time.sleep(0.5)
        else:
            process.terminate()
            raise RuntimeError("Streamlit did not become ready at localhost:8501.")

    yield

    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture
def dashboard(page: Page) -> Page:
    """Open the dashboard and load a public-safe output folder."""
    output_dir = PRIMARY_OUTPUT if PRIMARY_OUTPUT.exists() else FALLBACK_OUTPUT
    page.goto(APP_URL, wait_until="domcontentloaded")
    page.get_by_text("Load existing results", exact=True).first.click()
    page.get_by_label("Output directory").fill(str(output_dir))
    page.get_by_role("button", name="Load results").click()
    expect(
        page.get_by_role("heading", name="Step 1: Load and validate SMILES")
    ).to_be_visible(
        timeout=30_000
    )
    return page


def test_dashboard_has_readable_workflow_sections(dashboard: Page) -> None:
    """Verify the main workflow loads with readable presentation labels."""
    for text in (
        "Step 1: Load and validate SMILES",
        "Input used",
        "Output file created",
        "Valid and invalid SMILES",
    ):
        expect(dashboard.get_by_text(text, exact=True).first).to_be_visible()

    body = dashboard.locator("body")
    for internal_label in (
        "prioritization_score_with_nlp",
        "tanimoto_similarity",
        "best_reference_name",
        "surechembl_query_status",
    ):
        expect(body).not_to_contain_text(internal_label)


def test_molecule_detail_is_rendered_without_debug_output(dashboard: Page) -> None:
    """Verify the first workflow step is visual and table-based."""
    expect(dashboard.get_by_text("Workflow progress: 1 of 8")).to_be_visible()
    expect(
        dashboard.get_by_role(
            "button", name="Continue to Step 2: Chemical identity"
        )
    ).to_be_visible()

    body_text = dashboard.locator("body").inner_text()
    assert "Traceback" not in body_text
    assert "Uncaught app exception" not in body_text
    assert "'pubchem_status':" not in body_text
    assert '{"molecule_id"' not in body_text
