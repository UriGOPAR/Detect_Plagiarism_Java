import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========  
# Rutas de tus archivos  
# ==========  
test_pairs_path = 'C:/Users/urigo/Documents/Proyecto_Peter/versions/train_pairs.csv'
labels_path = 'C:/Users/urigo/Documents/Proyecto_Peter/versions/labels.csv'
predicciones_path = 'C:/Users/urigo/Documents/Proyecto_Peter/predictions_test_pairs.csv'   # Cambia ruta si está en otro lugar

# ==========================  
# Paso 1: Cargar archivos  
# ==========================  
# Cargar test_pairs
with open(test_pairs_path, 'r') as f:
    test_pairs = f.read().splitlines()

# Cargar labels
labels_df = pd.read_csv(labels_path)

# Cargar predicciones
pred_df = pd.read_csv(predicciones_path)

# ==========================  
# Paso 2: Crear un diccionario con las etiquetas verdaderas  
# ==========================  
# Creamos una columna "pair_folder" en el mismo formato que tu test_pairs
labels_df['pair_folder'] = labels_df['sub1'] + '_' + labels_df['sub2']
labels_dict = dict(zip(labels_df['pair_folder'], labels_df['verdict']))

# ==========================  
# Paso 3: Construir listas de etiquetas verdaderas y predichas SOLO para los pares del test  
# ==========================  
true_labels = []
pred_labels = []
pares_usados = []

for pair in test_pairs:
    if pair in labels_dict:
        true_label = labels_dict[pair]
        # Buscar predicción para este par
        pred_row = pred_df[pred_df['pair_folder'] == pair]
        if not pred_row.empty:
            pred_label = int(pred_row['prediction'].values[0])
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            pares_usados.append(pair)
        else:
            print(f"Advertencia: No hay predicción para {pair}")
    else:
        print(f"Advertencia: No se encontró etiqueta para {pair}")

# ==========================  
# Paso 4: Calcular métricas  
# ==========================  
accuracy = accuracy_score(true_labels, pred_labels)
cm = confusion_matrix(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, target_names=['No Plagio', 'Plagio'])

print("====== Resultados ======")
print(f"Total pares evaluados: {len(pares_usados)}")
print(f"Accuracy: {accuracy:.4f}")
print("\nMatriz de confusión:")
print(cm)
print("\nReporte de clasificación:")
print(report)
