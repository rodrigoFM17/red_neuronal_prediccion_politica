import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# 1. Carga y preprocesamiento de datos
def load_and_prepare_data(file_path):
    """
    Carga el dataset y realiza el preprocesamiento inicial
    """
    # Cargar dataset
    df = pd.read_csv(file_path, delimiter="|")
    
    # Verificar datos
    print(f"Forma del dataset: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")
    print(f"Distribución de clases:\n{df['clasificacion'].value_counts()}")
    
    # Manejar valores nulos
    df['titulo'].fillna('', inplace=True)
    df['contenido'].fillna('', inplace=True)
    
    # Combinar título y contenido
    df['texto_completo'] = df['titulo'] + " " + df['contenido']
    
    return df

# 2. Limpiar texto
def clean_text(text):
    """
    Limpia el texto: elimina caracteres especiales, convierte a minúsculas
    """
    if pd.isna(text):
        return ""
    
    # Convertir a texto si no lo es
    text = str(text)
    
    # Eliminar HTML
    text = re.sub(r'<.*?>', '', text)
    
    # Eliminar URLs
    text = re.sub(r'http\S+', '', text)
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Convertir a minúsculas
    return text.lower()

# 3. Añadir características estructuradas al texto
def add_structured_features(df):
    """
    Añade características estructuradas al texto para mejorar la clasificación
    """
    # Características que podrían ser útiles para determinar orientación política
    features = [
        'numero fuentes', 
        'estadisticas', 
        'numero adjetivos',
        'terminos ideologicos', 
        'citas directas', 
        'formalidad',
        'emocionalidad'
    ]
    
    # Para cada fila, añadir información estructurada al texto
    for idx, row in df.iterrows():
        additional_text = ""
        for feature in features:
            if feature in df.columns and not pd.isna(row[feature]):
                # Añadir característica como texto
                feature_name = feature.replace(" ", "_")
                value = row[feature]
                if isinstance(value, (int, float)) and value > 0:
                    # Repetir términos importantes según su valor
                    repeat = min(int(value), 5)  # Limitar repetición
                    additional_text += f" {feature_name} " * repeat
        
        # Añadir fuente y tema principal si están disponibles
        if 'fuente' in df.columns and not pd.isna(row['fuente']):
            additional_text += f" fuente_{row['fuente'].replace(' ', '_')}"
            
        if 'tema principal' in df.columns and not pd.isna(row['tema principal']):
            additional_text += f" tema_{row['tema principal'].replace(' ', '_')}"
            
        # Actualizar texto completo
        df.at[idx, 'texto_completo'] += additional_text
    
    return df

# 4. Construir modelo Naive Bayes
def build_naive_bayes_model():
    """
    Construye un pipeline con TF-IDF y Naive Bayes Multinomial
    """
    # Construir pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=5000,  # Limitar número de características
            min_df=2,           # Ignorar términos que aparecen en menos de 2 documentos
            max_df=0.8,         # Ignorar términos que aparecen en más del 80% de los documentos
            ngram_range=(1, 2), # Incluir unigramas y bigramas
            sublinear_tf=True   # Aplicar escalado sublineal a term frequency
        )),
        ('classifier', MultinomialNB(alpha=0.1))  # Laplace smoothing
    ])
    
    return pipeline

# 5. Validación cruzada
def cross_validate_model(model, X, y, n_folds=5):
    """
    Realiza validación cruzada y muestra resultados
    """
    # Configurar KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Métricas para cada fold
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'confusion_matrices': [],
        'best_model': None,
        'best_score': 0.0
    }
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nIniciando validación cruzada con {n_folds} folds...")
    
    fold_no = 1
    for train_index, test_index in kf.split(X):
        print(f"\nEntrenando fold {fold_no}/{n_folds}")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # Reiniciar modelo para este fold
        model.fit(X_train, y_train)
        
        # Predecir
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Guardar resultados
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1'].append(f1)
        results['confusion_matrices'].append(cm)
        
        print(f"Fold {fold_no} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Si este es el mejor modelo hasta ahora, guardarlo
        if f1 > results['best_score']:
            results['best_score'] = f1
            results['best_model'] = model
        
        fold_no += 1
    
    # Calcular y mostrar resultados promedio
    print("\nResultados promedio de validación cruzada:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    
    return results, label_encoder

# 6. Visualizar resultados
def plot_cross_validation_results(results):
    """
    Visualiza los resultados de la validación cruzada mostrando la precisión por cada fold
    """
    # Gráfico de barras para precisión por fold
    plt.figure(figsize=(12, 6))
    
    # Preparar datos
    n_folds = len(results['precision'])
    fold_numbers = [f"Fold {i+1}" for i in range(n_folds)]
    precision_values = results['precision']
    
    # Crear barras
    bars = plt.bar(fold_numbers, precision_values, alpha=0.7, color='skyblue')
    
    # Añadir etiquetas de valores
    for bar, val in zip(bars, precision_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{val:.4f}', ha='center', va='bottom')
    
    # Añadir línea de precisión promedio
    avg_precision = np.mean(precision_values)
    plt.axhline(y=avg_precision, color='red', linestyle='--', label=f'Promedio: {avg_precision:.4f}')
    
    # Etiquetas y título
    plt.title('Precisión por Fold en Validación Cruzada')
    plt.ylabel('Precisión')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Gráfico de todas las métricas por fold
    plt.figure(figsize=(14, 8))
    
    # Establecer ancho de barras
    n_metrics = 4
    width = 0.2
    x = np.arange(n_folds)
    
    # Crear barras para cada métrica
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['skyblue', 'lightgreen', 'salmon', 'violet']
    
    for i, metric in enumerate(metrics):
        values = results[metric]
        plt.bar(x + (i - n_metrics/2 + 0.5) * width, values, width, label=metric.capitalize(), color=colors[i])
    
    # Añadir etiquetas y título
    plt.xlabel('Fold')
    plt.ylabel('Puntuación')
    plt.title('Métricas por Fold en Validación Cruzada')
    plt.xticks(x, fold_numbers)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
# 7. Predecir con el modelo entrenado
def predict_political_orientation(model, label_encoder, text):
    """
    Predice la orientación política de un nuevo texto
    """
    # Limpiar el texto
    clean = clean_text(text)
    
    # Predecir
    prediction = model.predict_proba([clean])[0]
    
    # Obtener etiquetas y probabilidades
    indices = np.argsort(prediction)[::-1]
    labels = label_encoder.inverse_transform(indices)
    probs = prediction[indices]
    
    # Preparar resultados
    predicted_label = labels[0]
    confidence = float(probs[0])
    
    # Obtener top 3 alternativas
    top_3_labels = labels[:3]
    top_3_probs = probs[:3]
    
    alternative_predictions = [(label, float(prob)) for label, prob in zip(top_3_labels, top_3_probs)]
    
    return predicted_label, confidence, alternative_predictions

# 8. Función principal
def main(file_path):
    # Cargar y preparar datos
    df = load_and_prepare_data(file_path)
    
    # Limpiar textos
    df['texto_completo'] = df['texto_completo'].apply(clean_text)
    
    # Añadir características estructuradas al texto
    df = add_structured_features(df)
    
    # Construir modelo
    model = build_naive_bayes_model()
    
    # Preparar datos para validación cruzada
    X = df['texto_completo'].values
    y = df['clasificacion'].values
    
    # Realizar validación cruzada
    results, label_encoder = cross_validate_model(model, X, y, n_folds=5)
    
    # Visualizar resultados
    plot_cross_validation_results(results)
    
    # Entrenar modelo final con todos los datos
    final_model = build_naive_bayes_model()
    final_model.fit(X, y)
    
    # Guardar modelo y artefactos
    joblib.dump(final_model, 'modelo_naive_bayes.joblib')
    
    with open('label_encoder_nb.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle)
    
    print("Modelo y artefactos guardados correctamente.")
    
    # Ejemplo de predicción
    ejemplo_texto = """
"Gobierno de Brasil presenta reforma educativa para reducir desigualdades en el acceso a la educación superior"

Brasilia, 5 de agosto de 2025 — El presidente Luis Ignacio da Silva ha presentado una nueva reforma educativa destinada a reducir las brechas de acceso a la educación superior en Brasil. La reforma, denominada “Educación para Todos”, tiene como objetivo garantizar que los estudiantes de sectores más pobres puedan acceder a universidades públicas sin importar su origen socioeconómico, y mejorar la calidad educativa en las regiones más desatendidas.

La reforma propone aumentar las becas para los estudiantes de familias de bajos ingresos, además de crear nuevos programas de capacitación y desarrollo en las áreas rurales y en las comunidades indígenas. Asimismo, se priorizarán políticas de inclusión para mujeres y personas transgénero en las universidades, buscando reducir la discriminación y fomentar la igualdad de oportunidades para todos.

Sérgio Santos, Ministro de Educación, afirmó durante la presentación de la reforma: “Es hora de que nuestro sistema educativo sea verdaderamente inclusivo y garantice que ningún niño o joven quede atrás por razones de pobreza o discriminación. Esta reforma busca ofrecer una educación igualitaria para todos los brasileños”.

La reforma también incluye la ampliación de la infraestructura educativa en las regiones del norte y noreste del país, que históricamente han sufrido bajas tasas de escolarización. El gobierno espera que con estas medidas, el acceso a la educación superior sea más equitativo y que Brasil pueda avanzar hacia una sociedad más justa y solidaria.

    """
    
    predicted_label, confidence, alternatives = predict_political_orientation(
        final_model, label_encoder, ejemplo_texto
    )
    
    print(f"\nPredicción para texto de ejemplo:")
    print(f"Clasificación: {predicted_label} (confianza: {confidence:.2f})")
    print("Alternativas:")
    for alt_label, alt_conf in alternatives:
        print(f"- {alt_label}: {alt_conf:.2f}")
    
    return final_model, label_encoder

# Si se ejecuta como script principal
if __name__ == "__main__":
    # Reemplazar con la ruta de tu dataset
    file_path = "dataset_noticias_politicas.csv"
    main(file_path)