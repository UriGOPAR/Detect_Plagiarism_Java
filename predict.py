import os
import pandas as pd
import javalang
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ================================
# CONFIG
# ================================
dataset_root = './'
test_pairs_path = os.path.join(dataset_root, 'C:/Users/urigo/Documents/Proyecto_Peter/versions/train_pairs.csv')
version1_path = os.path.join(dataset_root, 'C:/Users/urigo/Documents/Proyecto_Peter/versions/version_1')

# Rutas a modelo y vectorizador entrenados
model_path = 'randomforest_qgrammar_model.pkl'
vectorizer_path = 'tfidf_vectorizer_qgrammar.pkl'

# ================================
# FUNCIONES DE PREPROCESAMIENTO
# ================================
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

def process_pair(pair_folder, version1_path):
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

    return pair_folder, qg1, qg2, f1, f2

# ================================
# MAIN PREDICTION SCRIPT
# ================================
if __name__ == "__main__":
    # Cargar modelo y vectorizador
    print("Cargando modelo y vectorizador...")
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)

    # Leer pares de test
    with open(test_pairs_path, 'r') as f:
        pairs = f.read().splitlines()

    processed_data = []
    print("Procesando pares de prueba...")
    for pair in pairs:
        result = process_pair(pair, version1_path)
        if result:
            processed_data.append(result)

    if not processed_data:
        print("No se procesaron pares correctamente.")
        exit()

    df = pd.DataFrame(processed_data, columns=['pair_folder', 'code1', 'code2', 'feat1', 'feat2'])

    # Vectorizar Q-grammar y calcular similitud coseno
    vec1 = tfidf.transform(df['code1']).toarray()
    vec2 = tfidf.transform(df['code2']).toarray()

    cos_sims = [cosine_similarity([v1], [v2])[0][0] for v1, v2 in zip(vec1, vec2)]
    cos_sims = np.array(cos_sims).reshape(-1, 1)

    feat1 = np.array(df['feat1'].tolist())
    feat2 = np.array(df['feat2'].tolist())

    X = np.hstack([vec1, vec2, cos_sims, feat1, feat2]).astype(np.float32)

    # Hacer predicciones
    print("Realizando predicciones...")
    preds = model.predict(X)

    # Guardar predicciones
    df_result = pd.DataFrame({
        'pair_folder': df['pair_folder'],
        'prediction': preds
    })

    output_path = 'predictions_test_pairs.csv'
    df_result.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}")
