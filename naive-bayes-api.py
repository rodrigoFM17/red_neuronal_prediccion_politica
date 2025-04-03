from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import re
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde otros dominios

# 1. Cargar modelo y artefactos
def load_model_artifacts():
    try:
        # Cargar modelo
        model = joblib.load('modelo_naive_bayes.joblib')
        
        # Cargar label encoder
        with open('label_encoder_nb.pickle', 'rb') as handle:
            label_encoder = pickle.load(handle)
            
        print("Modelo y artefactos cargados correctamente")
        return model, label_encoder
    except Exception as e:
        print(f"Error al cargar modelo y artefactos: {e}")
        return None, None

# 2. Función para limpiar texto
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
def add_structured_features_to_text(text, structured_data=None):
    """
    Añade características estructuradas al texto para mejorar la predicción
    """
    if structured_data is None:
        return text
    
    additional_text = ""
    
    # Características numéricas
    numeric_features = [
        'numero_fuentes', 'estadisticas', 'numero_adjetivos',
        'terminos_ideologicos', 'numero_palabras', 'imagenes', 
        'citas_directas'
    ]
    
    for feature in numeric_features:
        if feature in structured_data and structured_data[feature] > 0:
            # Repetir términos importantes según su valor
            repeat = min(int(structured_data[feature]), 5)  # Limitar repetición
            additional_text += f" {feature} " * repeat
    
    # Características booleanas
    boolean_features = [
        'medio_reconocido', 'medio_especializado', 'formalidad', 'emocionalidad'
    ]
    
    for feature in boolean_features:
        if feature in structured_data and structured_data[feature] == 1:
            additional_text += f" {feature}"
    
    # Características categóricas
    if 'fuente' in structured_data and structured_data['fuente']:
        additional_text += f" fuente_{structured_data['fuente'].replace(' ', '_')}"
    
    if 'tema_principal' in structured_data and structured_data['tema_principal']:
        additional_text += f" tema_{structured_data['tema_principal'].replace(' ', '_')}"
    
    if 'pais_de_origen' in structured_data and structured_data['pais_de_origen']:
        additional_text += f" pais_{structured_data['pais_de_origen'].replace(' ', '_')}"
    
    return text + additional_text

# 4. Predicción de orientación política
def predict_political_orientation(model, label_encoder, text, structured_data=None):
    """
    Predice la orientación política de un nuevo texto
    """
    # Limpiar el texto
    clean = clean_text(text)
    
    # Añadir características estructuradas
    processed_text = add_structured_features_to_text(clean, structured_data)
    
    # Predecir
    prediction = model.predict_proba([processed_text])[0]
    
    # Obtener etiquetas y probabilidades
    indices = np.argsort(prediction)[::-1]
    labels = label_encoder.inverse_transform(indices)
    probs = prediction[indices]
    
    # Preparar resultados
    results = []
    for i in range(min(len(labels), 5)):  # Top 5 resultados
        results.append({
            "categoria": labels[i],
            "probabilidad": float(probs[i])
        })
    
    return results

# Cargar modelo y artefactos al iniciar
model, label_encoder = load_model_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({"error": "El modelo no se ha cargado correctamente"}), 500
    
    # Obtener datos del request
    data = request.json
    
    if not data or 'texto' not in data:
        return jsonify({"error": "No se proporcionó el texto"}), 400
    
    # Extraer texto y datos estructurados
    text = data.get('texto', '')
    structured_data = data.get('datos_estructurados', None)
    
    try:
        # Obtener predicciones
        results = predict_political_orientation(model, label_encoder, text, structured_data)
        
        # Formar respuesta
        response = {
            "prediccion_principal": results[0],
            "alternativas": results[1:],
            "texto_analizado": text[:100] + "..." if len(text) > 100 else text
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if model is None or label_encoder is None:
        return jsonify({"status": "error", "message": "Modelo no cargado"}), 500
    
    return jsonify({"status": "ok", "message": "API funcionando correctamente"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
