import os
import pandas as pd
import pytest
from src.utils import load_data

# Creamos un CSV de prueba temporal
@pytest.fixture
def sample_csv(tmp_path):
    file_path = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "title": ["Paper A", "Paper B"],
        "abstract": ["This is about medicine", "This is about biology"],
        "group": ["Medicina", "Biología"]
    })
    df.to_csv(file_path, index=False)
    return file_path

def test_load_data(sample_csv):
    """Prueba que load_data cargue el CSV correctamente"""
    df = load_data(sample_csv)

    # Verificar columnas esperadas
    assert "title" in df.columns
    assert "abstract" in df.columns
    assert "group" in df.columns
    assert "text" in df.columns

    # Verificar concatenación de texto
    assert df.loc[0, "text"] == "Paper A This is about medicine"
    assert df.loc[1, "text"] == "Paper B This is about biology"

def test_missing_columns(tmp_path):
    """Prueba que load_data falle si faltan columnas"""
    file_path = tmp_path / "bad.csv"
    pd.DataFrame({"title": ["Only title"]}).to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        load_data(file_path)

