from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import re

app = Flask(__name__)
CORS(app) 

def load_model_artifacts():
    try:
        model = tf.keras.models.load_model('modelo_red_neuronal.h5')
        
        with open('vectorizer.pickle', 'rb') as handle:
            vectorizer = pickle.load(handle)
        
        with open('label_encoder.pickle', 'rb') as handle:
            label_encoder = pickle.load(handle)
        
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

def clean_text(text):
  
    if text is None:
        return ""
    
    text = str(text)
    
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'http\S+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text.lower()

def prepare_structured_features(structured_data, scaler):
    
    if structured_data is None or scaler is None:
        return None
    
    try:
        numeric_features = [
            'numero_fuentes', 'estadisticas', 'numero_adjetivos',
            'terminos_ideologicos', 'numero_palabras', 'imagenes', 
            'citas_directas'
        ]
        
        boolean_features = [
            'medio_reconocido', 'medio_especializado', 'formalidad', 'emocionalidad'
        ]
        
        numeric_values = []
        for feature in numeric_features:
            if feature in structured_data:
                numeric_values.append(float(structured_data[feature]))
            else:
                numeric_values.append(0.0)
        
        boolean_values = []
        for feature in boolean_features:
            if feature in structured_data:
                boolean_values.append(float(structured_data[feature]))
            else:
                boolean_values.append(0.0)
        
        features = np.array([numeric_values + boolean_values])
        
        scaled_features = scaler.transform(features)
        
        return scaled_features
    except Exception as e:
        print(f"Error preparando características estructuradas: {e}")
        return None

def predict_political_orientation(model, vectorizer, label_encoder, text, structured_data=None, scaler=None):
    
    try:
        clean = clean_text(text)
        X_text = vectorizer.transform([clean]).toarray()
        
        X_struct = prepare_structured_features(structured_data, scaler)
        
        if isinstance(model.input, list):
            if X_struct is not None:
                prediction = model.predict([X_text, X_struct])
            else:
                dummy_struct = np.zeros((1, model.input[1].shape[1]))
                prediction = model.predict([X_text, dummy_struct])
        else:
            prediction = model.predict(X_text)
        
        indices = np.argsort(prediction[0])[-5:][::-1] 
        labels = label_encoder.inverse_transform(indices)
        probs = prediction[0][indices]
        
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

model, vectorizer, label_encoder, scaler = load_model_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None or label_encoder is None:
        return jsonify({"error": "El modelo no se ha cargado correctamente"}), 500
    
    data = request.json
    
    if not data or 'texto' not in data:
        return jsonify({"error": "No se proporcionó el texto"}), 400
    
    text = data.get('texto', '')
    structured_data = data.get('datos_estructurados', None)
    
    try:
        results = predict_political_orientation(
            model, vectorizer, label_encoder, text, structured_data, scaler
        )
        
        if results is None:
            return jsonify({"error": "Error al procesar la predicción"}), 500
        
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
    
    if model is None:
        return jsonify({"error": "El modelo no se ha cargado correctamente"}), 500
    
    try:
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        
        layers_info = []
        for i, layer in enumerate(model.layers):
            layer_info = {
                "nombre": layer.name,
                "tipo": layer.__class__.__name__,
                "neuronas": layer.output_shape,
                "activacion": layer.get_config().get('activation', 'None') if hasattr(layer, 'get_config') else 'None'
            }
            layers_info.append(layer_info)
        
        vectorizer_info = {
            "max_features": getattr(vectorizer, 'max_features', 'Unknown'),
            "ngram_range": getattr(vectorizer, 'ngram_range', 'Unknown'),
            "vocabulary_size": len(getattr(vectorizer, 'vocabulary_', {}))
        }
        
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