import os
import pandas as pd
import javalang
import re
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

# ================
# Rutas
# ================
pairs_path = 'C:/Users/urigo/Documents/Proyecto_Peter/versions/train_pairs.csv'
version1_path = 'C:/Users/urigo/Documents/Proyecto_Peter/versions/version_1'
model_path = 'plagiarism_model.pth'
vectorizer_path = 'tfidf_vectorizer.pkl'

# =========================
# Clase del modelo (igual que antes)
# =========================
class PlagiarismClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.drop1(torch.relu(self.bn1(self.fc1(x))))
        x = self.drop2(torch.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

# =========================
# Preprocesamiento (igual que antes)
# =========================
class IdentifierNormalizer:
    def __init__(self):
        self.counter = 0
        self.mapping = {}

    def normalize(self, name):
        if name not in self.mapping:
            self.counter += 1
            self.mapping[name] = f'id{self.counter}'
        return self.mapping[name]

def extract_normalized_code(code):
    try:
        tree = javalang.parse.parse(code)
    except:
        return ''
    normalizer = IdentifierNormalizer()
    tokens = []
    for path, node in tree:
        node_type = type(node).__name__
        if hasattr(node, 'name') and isinstance(node.name, str):
            normalized = normalizer.normalize(node.name)
            tokens.append(f'{node_type}:{normalized}')
        elif hasattr(node, 'value') and isinstance(node.value, str):
            tokens.append(f'{node_type}:{node.value}')
        else:
            tokens.append(node_type)
    return ' '.join(tokens)

def remove_comments_and_whitespace(code):
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return code

def process_pair(pair_folder):
    folder_path = os.path.join(version1_path, pair_folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.java')]
    if len(files) != 2:
        print(f"Advertencia: {pair_folder} no contiene exactamente 2 archivos Java")
        return None
    file1_path = os.path.join(folder_path, files[0])
    file2_path = os.path.join(folder_path, files[1])

    with open(file1_path, 'r', encoding='utf8') as f:
        code1 = f.read()
    with open(file2_path, 'r', encoding='utf8') as f:
        code2 = f.read()

    code1 = remove_comments_and_whitespace(code1)
    code2 = remove_comments_and_whitespace(code2)

    norm_code1 = extract_normalized_code(code1)
    norm_code2 = extract_normalized_code(code2)

    return norm_code1, norm_code2

# =========================
# Predicción
# =========================
def predict_pairs(pairs_csv_path, output_csv_path='predicciones.csv'):
    # Cargar vectorizador y modelo
    tfidf_vectorizer = joblib.load(vectorizer_path)
    input_dim = 2 * len(tfidf_vectorizer.vocabulary_) + 1
    model = PlagiarismClassifier(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Leer nombres de pares
    with open(pairs_csv_path, 'r') as f:
        pairs = f.read().splitlines()

    predictions = []

    for pair in pairs:
        result = process_pair(pair)
        if result is None:
            continue
        norm_code1, norm_code2 = result

        vec1 = tfidf_vectorizer.transform([norm_code1]).toarray()
        vec2 = tfidf_vectorizer.transform([norm_code2]).toarray()
        cos_sim = cosine_similarity(vec1, vec2)[0][0]
        features = np.hstack([vec1, vec2, [[cos_sim]]]).astype(np.float32)

        with torch.no_grad():
            inputs = torch.from_numpy(features)
            output = torch.sigmoid(model(inputs)).item()
            pred_label = 1 if output > 0.5 else 0

        predictions.append((pair, pred_label))

    # Guardar predicciones en CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pair_name', 'prediction'])
        writer.writerows(predictions)

    print(f"Predicciones guardadas en {output_csv_path}")

# =========================
# Ejecutar
# =========================
if __name__ == "__main__":
    predict_pairs(pairs_path)
