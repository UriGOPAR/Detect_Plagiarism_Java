import os
import pandas as pd
import javalang
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader

# ================================
# CONFIG
# ================================
dataset_root = './'
pairs_path = os.path.join(dataset_root, 'C:/Users/urigo/Documents/Proyecto_Peter/versions/test_pairs.csv')
version1_path = os.path.join(dataset_root, 'C:/Users/urigo/Documents/Proyecto_Peter/versions/version_1')
labels_path = os.path.join(dataset_root, 'C:/Users/urigo/Documents/Proyecto_Peter/versions/labels.csv')
batch_size = 16
epochs = 30
learning_rate = 0.001

# ================================
# PREPROCESAMIENTO DEL CÓDIGO JAVA
# ================================
df_labels = pd.read_csv(labels_path)
label_dict = {f"{row['sub1']}_{row['sub2']}": row['verdict'] for _, row in df_labels.iterrows()}

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

    label = label_dict.get(pair_folder, None)
    if label is None:
        print(f"Etiqueta no encontrada para {pair_folder}")
        return None

    return norm_code1, norm_code2, label

def process_all_pairs():
    dataset = []
    with open(pairs_path, 'r') as f:
        pairs = f.read().splitlines()

    for pair in pairs:
        result = process_pair(pair)
        if result:
            dataset.append(result)

    return dataset

# ================================
# MODELO PYTORCH CON pos_weight
# ================================
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
        x = self.fc3(x)   # No sigmoid aquí
        return x

# ================================
# MAIN PIPELINE
# ================================
if __name__ == "__main__":
    dataset = process_all_pairs()
    print(f"Total pares procesados: {len(dataset)}")

    df = pd.DataFrame(dataset, columns=['code1', 'code2', 'label'])
    df.to_csv('preprocessed_dataset.csv', index=False)
    print("Dataset preprocesado guardado en preprocessed_dataset.csv")

    print("Generando embeddings TF-IDF...")
    all_code = pd.concat([df['code1'], df['code2']])
    tfidf_vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_vectorizer.fit(all_code)

    vectors_code1 = tfidf_vectorizer.transform(df['code1']).toarray()
    vectors_code2 = tfidf_vectorizer.transform(df['code2']).toarray()

    cos_sims = []
    for v1, v2 in zip(vectors_code1, vectors_code2):
        sim = cosine_similarity([v1], [v2])[0][0]
        cos_sims.append(sim)

    cos_sims = np.array(cos_sims).reshape(-1, 1)
    X = np.hstack([vectors_code1, vectors_code2, cos_sims]).astype(np.float32)
    y = df['label'].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = PlagiarismClassifier(input_dim=X.shape[1])

    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = torch.sigmoid(model(inputs))
            preds = (outputs > 0.5).float()
            preds_list.extend(preds.numpy().flatten())
            targets_list.extend(targets.numpy().flatten())

    acc = np.mean(np.array(preds_list) == np.array(targets_list))
    print(f"Exactitud en test: {acc * 100:.2f}%")

    cm = confusion_matrix(targets_list, preds_list)
    print("Matriz de confusión:")
    print(cm)

    print("\nReporte de clasificación:")
    print(classification_report(targets_list, preds_list, digits=4))
    
    import joblib

# Guardar modelo
torch.save(model.state_dict(), 'plagiarism_model.pth')
print("Modelo guardado en plagiarism_model.pth")

# Guardar vectorizador TF-IDF
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print("Vectorizador TF-IDF guardado en tfidf_vectorizer.pkl")

# =========================
# Graficar matriz de confusión
# =========================
plt.figure(figsize=(6,5))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.xlabel('Predicción', fontsize=14)
plt.ylabel('Etiqueta Real', fontsize=14)
plt.title('Matriz de Confusión', fontsize=16)
plt.xticks([0.5, 1.5], ['No Plagio (0)', 'Plagio (1)'], fontsize=12)
plt.yticks([0.5, 1.5], ['No Plagio (0)', 'Plagio (1)'], fontsize=12, rotation=0)
plt.tight_layout()
plt.show()
plt.savefig('confusion_matrix.png')
print("Matriz de confusión guardada como confusion_matrix.png")


