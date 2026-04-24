import pandas as pd
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Carregar datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Criar rótulos
fake["label"] = 0
true["label"] = 1

# Juntar dados
data = pd.concat([fake, true])

# Selecionar colunas
data = data[["text", "label"]]

# Limpeza básica
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

data["text"] = data["text"].apply(clean_text)

# Separar treino e teste
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vetorização
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Modelo
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Previsão
y_pred = model.predict(X_test_vec)

# Métricas
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Salvar modelo
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Modelo salvo com sucesso!")
