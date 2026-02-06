from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def prompt_single_path():
    return str(FIXTURES_DIR / "prompt_single.txt")


@pytest.fixture
def prompt_table_path():
    return str(FIXTURES_DIR / "prompt_table.txt")


@pytest.fixture
def prompt_bad_path():
    return str(FIXTURES_DIR / "prompt_bad.txt")


@pytest.fixture
def csv_single_path():
    return str(FIXTURES_DIR / "test_single.csv")


@pytest.fixture
def csv_table_path():
    return str(FIXTURES_DIR / "test_table.csv")
