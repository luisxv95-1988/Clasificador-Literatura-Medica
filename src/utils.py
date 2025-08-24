import pandas as pd

def load_data(path):
    """Carga el dataset desde CSV con columnas title, abstract, group"""
    df = pd.read_csv(path)
    if not {"title", "abstract", "group"}.issubset(df.columns):
        raise ValueError("El CSV debe contener las columnas: title, abstract, group")
    df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
    return df

