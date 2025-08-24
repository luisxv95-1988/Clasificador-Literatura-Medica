import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from utils import load_data

def evaluate(data_path="data/challenge_data-18-ago.csv", model_path="models/classifier.joblib", vectorizer_path="models/vectorizer.joblib"):
    """Evalúa el modelo guardado en disco"""
    df = load_data(data_path)

    X = df["text"]
    y_true = df["group"]

    # Cargar modelo y vectorizador
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    X_tfidf = vectorizer.transform(X)
    y_pred = model.predict(X_tfidf)

    print("Reporte de clasificación:\n", classification_report(y_true, y_pred))
    print("F1-score ponderado:", f1_score(y_true, y_pred, average="weighted"))
    print("Matriz de confusión:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()

