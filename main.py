import os
import pandas as pd
import javalang
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib

# ================================
# CONFIG
# ================================
dataset_root = './'
pairs_path = os.path.join(dataset_root, 'C:/Users/urigo/Documents/Proyecto_Peter/versions/test_pairs.csv')
version1_path = os.path.join(dataset_root, 'C:/Users/urigo/Documents/Proyecto_Peter/versions/version_1')
labels_path = os.path.join(dataset_root, 'C:/Users/urigo/Documents/Proyecto_Peter/versions/labels.csv')

# ================================
# PREPROCESAMIENTO
# ================================
df_labels = pd.read_csv(labels_path)
label_dict = {f"{row['sub1']}_{row['sub2']}": row['verdict'] for _, row in df_labels.iterrows()}

def remove_comments_and_whitespace(code):
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return code

def extract_q_grammar(code, q=3):
    try:
        tree = javalang.parse.parse(code)
    except:
        return []

    paths = []

    def dfs(node, path):
        path.append(type(node).__name__)
        if len(path) == q:
            paths.append("->".join(path))
            path.pop(0)
        for child in node.children:
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, javalang.tree.Node):
                        dfs(item, path.copy())
            elif isinstance(child, javalang.tree.Node):
                dfs(child, path.copy())

    dfs(tree, [])
    return paths

def extract_code_features(code):
    num_lines = len(code.split('\n'))
    num_comments = len(re.findall(r"//|/\*|\*/", code))
    try:
        tree = javalang.parse.parse(code)
        num_classes = sum(isinstance(path, javalang.tree.ClassDeclaration) for path in tree.types)
        num_methods = sum(1 for _, node in tree.filter(javalang.tree.MethodDeclaration))
    except:
        num_classes = 0
        num_methods = 0

    return [num_lines, num_classes, num_methods, num_comments]

def process_pair(pair_folder):
    folder_path = os.path.join(version1_path, pair_folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.java')]

    if len(files) != 2:
        return None

    file1_path = os.path.join(folder_path, files[0])
    file2_path = os.path.join(folder_path, files[1])

    with open(file1_path, 'r', encoding='utf8') as f:
        code1 = f.read()
    with open(file2_path, 'r', encoding='utf8') as f:
        code2 = f.read()

    clean1 = remove_comments_and_whitespace(code1)
    clean2 = remove_comments_and_whitespace(code2)

    qg1 = " ".join(extract_q_grammar(clean1, q=3))
    qg2 = " ".join(extract_q_grammar(clean2, q=3))

    f1 = extract_code_features(code1)
    f2 = extract_code_features(code2)

    label = label_dict.get(pair_folder, None)
    if label is None:
        return None

    return qg1, qg2, f1, f2, label

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
# MAIN PIPELINE
# ================================
if __name__ == "__main__":
    dataset = process_all_pairs()
    print(f"Total pares procesados: {len(dataset)}")

    df = pd.DataFrame(dataset, columns=['code1', 'code2', 'feat1', 'feat2', 'label'])

    all_code = pd.concat([df['code1'], df['code2']])
    tfidf = TfidfVectorizer(max_features=3000)
    tfidf.fit(all_code)

    vec1 = tfidf.transform(df['code1']).toarray()
    vec2 = tfidf.transform(df['code2']).toarray()

    from sklearn.metrics.pairwise import cosine_similarity
    cos_sims = [cosine_similarity([v1], [v2])[0][0] for v1, v2 in zip(vec1, vec2)]
    cos_sims = np.array(cos_sims).reshape(-1, 1)

    feat1 = np.array(df['feat1'].tolist())
    feat2 = np.array(df['feat2'].tolist())

    X = np.hstack([vec1, vec2, cos_sims, feat1, feat2]).astype(np.float32)
    y = df['label'].astype(np.int32)

    print("Aplicando SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    print("Ajustando hiperparámetros con validación cruzada...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(rf, param_grid, cv=StratifiedKFold(5), scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Mejores parámetros: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    print("Evaluando modelo final...")
    y_pred = best_model.predict(X_test)

    acc = np.mean(y_pred == y_test)
    print(f"Exactitud en test: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(cm)

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(best_model, 'randomforest_qgrammar_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer_qgrammar.pkl')
    print("Modelo y vectorizador guardados")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.xticks([0.5, 1.5], ['No Plagio (0)', 'Plagio (1)'], fontsize=12)
    plt.yticks([0.5, 1.5], ['No Plagio (0)', 'Plagio (1)'], fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_rf.png')
    plt.show()
