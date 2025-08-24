import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from utils import load_data

def train_model(data_path="data/challenge_data-18-ago.csv", model_path="models/classifier.joblib", vectorizer_path="models/vectorizer.joblib"):
    """Entrena el modelo y lo guarda en disco"""
    df = load_data(data_path)

    X = df["text"]
    y = df["group"]

    # Separar train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorizador + modelo
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    # Evaluación preliminar
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print("F1-score ponderado:", f1_score(y_test, y_pred, average="weighted"))

    # Guardar modelo y vectorizador
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"✅ Modelo guardado en {model_path}")
    print(f"✅ Vectorizador guardado en {vectorizer_path}")

if __name__ == "__main__":
    train_model()

