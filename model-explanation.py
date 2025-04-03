import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

def load_model_and_data(model_path='modelo_naive_bayes.joblib', 
                       label_encoder_path='label_encoder_nb.pickle',
                       data_path='dataset_noticias_politicas.csv'):
    """
    Carga el modelo, label encoder y datos originales
    """
    # Cargar modelo
    model = joblib.load(model_path)
    
    # Cargar label encoder
    with open(label_encoder_path, 'rb') as handle:
        label_encoder = pickle.load(handle)
    
    # Cargar datos
    df = pd.read_csv(data_path)
    
    return model, label_encoder, df

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

def get_top_features_per_class(model, label_encoder, n_features=20):
    """
    Obtiene las características más importantes para cada clase
    """
    # Obtener vectorizador del pipeline
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']
    
    # Obtener feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Obtener coeficientes para cada clase
    feature_importance = {}
    
    for i, category in enumerate(label_encoder.classes_):
        # Para Naive Bayes Multinomial, los coeficientes son log-probabilidades
        coefs = classifier.feature_log_prob_[i]
        
        # Ordenar características por importancia
        top_indices = coefs.argsort()[::-1][:n_features]
        top_features = [(feature_names[j], coefs[j]) for j in top_indices]
        
        feature_importance[category] = top_features
    
    return feature_importance

def plot_top_features(feature_importance, n_classes=None, n_features=10):
    """
    Visualiza las características más importantes para cada clase
    """
    # Si n_classes no se especifica, mostrar todas las clases
    if n_classes is None:
        n_classes = len(feature_importance)
    
    # Seleccionar las primeras n_classes
    categories = list(feature_importance.keys())[:n_classes]
    
    # Crear figura con subplots
    fig, axes = plt.subplots(n_classes, 1, figsize=(12, n_classes * 4), sharex=True)
    
    # Si solo hay una clase, convertir axes en lista para consistencia
    if n_classes == 1:
        axes = [axes]
    
    for i, category in enumerate(categories):
        # Obtener las características más importantes
        top_features = feature_importance[category][:n_features]
        
        # Separar nombres y valores
        names = [feat[0] for feat in top_features]
        values = [feat[1] for feat in top_features]
        
        # Crear gráfico de barras
        bars = axes[i].barh(names, values, color='skyblue')
        
        # Añadir etiquetas de valores
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.02
            axes[i].text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                        va='center')
        
        # Título y etiquetas
        axes[i].set_title(f'Palabras clave para "{category}"')
        axes[i].set_xlabel('Log-Probabilidad')
    
    plt.tight_layout()
    plt.show()

def analyze_feature_presence(df, feature_importance, top_n=5):
    """
    Analiza la presencia de características importantes en cada documento
    """
    # Limpiar textos
    df['texto_limpio'] = df['titulo'].fillna('') + ' ' + df['contenido'].fillna('')
    df['texto_limpio'] = df['texto_limpio'].apply(clean_text)
    
    # Para cada clase, calcular la frecuencia de sus características principales
    feature_presence = {}
    
    for category, features in feature_importance.items():
        # Tomar las top_n características
        top_features = [feat[0] for feat in features[:top_n]]
        
        # Filtrar documentos de esta categoría
        category_docs = df[df['clasificacion'] == category]
        
        if len(category_docs) == 0:
            continue
        
        # Calcular presencia de cada característica
        presence = {}
        for feature in top_features:
            # Contar documentos que contienen esta característica
            count = sum(category_docs['texto_limpio'].str.contains(feature, case=False, regex=False))
            presence[feature] = count / len(category_docs)
        
        feature_presence[category] = presence
    
    # Visualizar resultados
    categories = list(feature_presence.keys())
    
    if not categories:
        print("No hay datos para analizar")
        return
    
    # Crear figura
    plt.figure(figsize=(14, 10))
    
    # Para cada categoría, graficar la presencia de características
    for i, category in enumerate(categories):
        presence = feature_presence[category]
        features = list(presence.keys())
        values = list(presence.values())
        
        # Posición de las barras
        x = np.arange(len(features))
        width = 0.8 / len(categories)
        
        plt.bar(x + i * width, values, width, label=category)
    
    # Etiquetas y configuración
    plt.xlabel('Características')
    plt.ylabel('Proporción de documentos')
    plt.title('Presencia de características principales en documentos por categoría')
    plt.xticks(x + width * (len(categories) - 1) / 2, features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_model_uncertainty(model, df):
    """
    Analiza la incertidumbre del modelo al clasificar diferentes textos
    """
    # Limpiar textos
    df['texto_limpio'] = df['titulo'].fillna('') + ' ' + df['contenido'].fillna('')
    df['texto_limpio'] = df['texto_limpio'].apply(clean_text)
    
    # Obtener predicciones
    probabilities = model.predict_proba(df['texto_limpio'])
    
    # Calcular incertidumbre (entropia de Shannon)
    def entropy(probs):
        return -np.sum(probs * np.log2(probs + 1e-10))  # Añadir pequeño valor para evitar log(0)
    
    entropies = [entropy(probs) for probs in probabilities]
    df['incertidumbre'] = entropies
    
    # Obtener clase predicha
    df['clase_predicha'] = model.predict(df['texto_limpio'])
    
    # Calcular confianza (probabilidad de la clase predicha)
    df['confianza'] = [max(probs) for probs in probabilities]
    
    # Crear gráfico de incertidumbre por clase real
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='clasificacion', y='incertidumbre', data=df)
    plt.title('Incertidumbre del modelo por clase real')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Crear gráfico de confianza por clase real
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='clasificacion', y='confianza', data=df)
    plt.title('Confianza del modelo por clase real')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Distribución de confianza por clase predicha vs real
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='clasificacion', y='confianza', hue='clase_predicha', data=df)
    plt.title('Confianza por clase real y predicha')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Clase predicha', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return df

def explain_prediction(model, text, label_encoder, n_features=10):
    """
    Explica una predicción específica mostrando la contribución de cada palabra
    """
    # Limpiar texto
    clean = clean_text(text)
    
    # Vectorizar el texto
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']
    
    # Convertir texto a vector TF-IDF
    X = vectorizer.transform([clean])
    
    # Obtener feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Obtener índices de características no cero
    feature_indices = X.nonzero()[1]
    
    # Si no hay características reconocidas, devolver mensaje
    if len(feature_indices) == 0:
        print("No se encontraron características reconocidas en el texto")