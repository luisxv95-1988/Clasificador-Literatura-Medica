import argparse
import pandas as pd
import joblib

def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict(input_csv, output_csv, model_path="models/classifier.joblib", vectorizer_path="models/vectorizer.joblib"):
    """Genera predicciones a partir de un CSV nuevo"""
    data = pd.read_csv(input_csv)

    if not {"title", "abstract"}.issubset(data.columns):
        raise ValueError("El CSV debe contener las columnas: title y abstract")

    data["text"] = data["title"].fillna("") + " " + data["abstract"].fillna("")

    model, vectorizer = load_model(model_path, vectorizer_path)
    X_new = vectorizer.transform(data["text"])
    preds = model.predict(X_new)

    data["group_predicted"] = preds
    data.to_csv(output_csv, index=False)
    print(f"✅ Predicciones guardadas en {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar predicciones con el clasificador médico")
    parser.add_argument("--data", type=str, required=True, help="Ruta del CSV de entrada con columnas title y abstract")
    parser.add_argument("--out", type=str, required=True, help="Ruta del archivo CSV de salida con predicciones")

    args = parser.parse_args()
    predict(args.data, args.out)
