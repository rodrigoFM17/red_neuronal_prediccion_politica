from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import re

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde otros dominios

# 1. Cargar modelo y artefactos
def load_model_artifacts():
    try:
        # Cargar modelo
        model = tf.keras.models.load_model('modelo_red_neuronal.h5')
        
        # Cargar vectorizador
        with open('vectorizer.pickle', 'rb') as handle:
            vectorizer = pickle.load(handle)
        
        # Cargar label encoder
        with open('label_encoder.pickle', 'rb') as handle:
            label_encoder = pickle.load(handle)
        
        # Cargar scaler si existe
        try:
            with open('scaler.pickle', 'rb') as handle:
                scaler = pickle.load(handle)
        except:
            scaler = None
        
        print("Modelo y artefactos cargados correctamente")
        return model, vectorizer, label_encoder, scaler
    except Exception as e:
        print(f"Error al cargar modelo y artefactos: {e}")
        return None, None, None, None

# 2. Limpiar texto
def clean_text(text):
    """
    Limpia el texto: elimina caracteres especiales, convierte a minúsculas
    """
    if text is None:
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

# 3. Preparar características estructuradas
def prepare_structured_features(structured_data, scaler):
    """
    Prepara las características estructuradas para el modelo
    """
    if structured_data is None or scaler is None:
        return None
    
    try:
        # Características numéricas
        numeric_features = [
            'numero_fuentes', 'estadisticas', 'numero_adjetivos',
            'terminos_ideologicos', 'numero_palabras', 'imagenes', 
            'citas_directas'
        ]
        
        # Características booleanas
        boolean_features = [
            'medio_reconocido', 'medio_especializado', 'formalidad', 'emocionalidad'
        ]
        
        # Extraer valores numéricos
        numeric_values = []
        for feature in numeric_features:
            if feature in structured_data:
                numeric_values.append(float(structured_data[feature]))
            else:
                numeric_values.append(0.0)
        
        # Extraer valores booleanos
        boolean_values = []
        for feature in boolean_features:
            if feature in structured_data:
                boolean_values.append(float(structured_data[feature]))
            else:
                boolean_values.append(0.0)
        
        # Combinar características
        features = np.array([numeric_values + boolean_values])
        
        # Escalar características numéricas
        scaled_features = scaler.transform(features)
        
        return scaled_features
    except Exception as e:
        print(f"Error preparando características estructuradas: {e}")
        return None

# 4. Predecir orientación política
def predict_political_orientation(model, vectorizer, label_encoder, text, structured_data=None, scaler=None):
    """
    Predice la orientación política de un texto
    """
    try:
        # Limpiar y vectorizar texto
        clean = clean_text(text)
        X_text = vectorizer.transform([clean]).toarray()
        
        # Preparar características estructuradas
        X_struct = prepare_structured_features(structured_data, scaler)
        
        # Verificar si el modelo espera múltiples entradas
        if isinstance(model.input, list):
            # Modelo con múltiples entradas (texto + estructuradas)
            if X_struct is not None:
                prediction = model.predict([X_text, X_struct])
            else:
                # Crear características estructuradas dummy
                dummy_struct = np.zeros((1, model.input[1].shape[1]))
                prediction = model.predict([X_text, dummy_struct])
        else:
            # Modelo con una sola entrada (solo texto)
            prediction = model.predict(X_text)
        
        # Obtener resultados
        indices = np.argsort(prediction[0])[-5:][::-1]  # Top 5 predicciones
        labels = label_encoder.inverse_transform(indices)
        probs = prediction[0][indices]
        
        # Formar lista de resultados
        results = []
        for i in range(len(labels)):
            results.append({
                "categoria": labels[i],
                "probabilidad": float(probs[i])
            })
        
        return results
    except Exception as e:
        print(f"Error al realizar predicción: {e}")
        return None

# Cargar modelo y artefactos al iniciar
model, vectorizer, label_encoder, scaler = load_model_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None or label_encoder is None:
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
        results = predict_political_orientation(
            model, vectorizer, label_encoder, text, structured_data, scaler
        )
        
        if results is None:
            return jsonify({"error": "Error al procesar la predicción"}), 500
        
        # Formar respuesta
        response = {
            "prediccion_principal": results[0],
            "alternativas": results[1:],
            "texto_analizado": text[:100] + "..." if len(text) > 100 else text
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Retorna información sobre el modelo
    """
    if model is None:
        return jsonify({"error": "El modelo no se ha cargado correctamente"}), 500
    
    try:
        # Obtener resumen del modelo
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        
        # Contar neuronas por capa
        layers_info = []
        for i, layer in enumerate(model.layers):
            layer_info = {
                "nombre": layer.name,
                "tipo": layer.__class__.__name__,
                "neuronas": layer.output_shape,
                "activacion": layer.get_config().get('activation', 'None') if hasattr(layer, 'get_config') else 'None'
            }
            layers_info.append(layer_info)
        
        # Información sobre vectorizador
        vectorizer_info = {
            "max_features": getattr(vectorizer, 'max_features', 'Unknown'),
            "ngram_range": getattr(vectorizer, 'ngram_range', 'Unknown'),
            "vocabulary_size": len(getattr(vectorizer, 'vocabulary_', {}))
        }
        
        # Formar respuesta
        response = {
            "tipo_modelo": "Red Neuronal Feed-Forward",
            "num_clases": len(label_encoder.classes_),
            "clases": label_encoder.classes_.tolist(),
            "capas": layers_info,
            "resumen": model_summary,
            "vectorizador": vectorizer_info
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if model is None or vectorizer is None or label_encoder is None:
        return jsonify({"status": "error", "message": "Modelo no cargado"}), 500
    
    return jsonify({
        "status": "ok", 
        "message": "API funcionando correctamente",
        "modelo": "Red Neuronal Feed-Forward",
        "clases": len(label_encoder.classes_)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)