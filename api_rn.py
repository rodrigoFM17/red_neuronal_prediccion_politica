from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde otros dominios

# Cargar modelo y artefactos (solo una vez al iniciar)
def load_artifacts():
    model = load_model('modelo_clasificacion_noticias.h5')
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
    
    try:
        with open('scaler.pickle', 'rb') as handle:
            scaler = pickle.load(handle)
    except:
        scaler = None
    
    try:
        with open('encoders.pickle', 'rb') as handle:
            encoders = pickle.load(handle)
    except:
        encoders = None
    
    return model, tokenizer, label_encoder, scaler, encoders

model, tokenizer, label_encoder, scaler, encoders = load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extraer texto
    text = data.get('texto', '')
    if not text:
        return jsonify({"error": "No se proporcionó texto"}), 400
    
    # Extraer datos estructurados si existen
    structured_data = data.get('datos_estructurados', None)
    
    # Preprocesar texto
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    
    # Verificar si es modelo híbrido
    is_hybrid_model = len(model.inputs) > 1
    
    try:
        if is_hybrid_model and structured_data:
            # Procesar datos estructurados
            # (Implementar según tus necesidades)
            
            # Simplificado: usar matriz de ceros
            dummy_data = np.zeros((1, model.inputs[1].shape[1]))
            prediction = model.predict([padded, dummy_data])[0]
        elif is_hybrid_model:
            # Sin datos estructurados pero modelo híbrido
            dummy_data = np.zeros((1, model.inputs[1].shape[1]))
            prediction = model.predict([padded, dummy_data])[0]
        else:
            # Modelo solo texto
            prediction = model.predict(padded)[0]
        
        # Obtener Top-3 predicciones
        top_indices = prediction.argsort()[-3:][::-1]
        top_labels = label_encoder.inverse_transform(top_indices)
        top_probs = prediction[top_indices].tolist()
        
        # Formar respuesta
        results = [
            {"categoria": label, "probabilidad": prob} 
            for label, prob in zip(top_labels, top_probs)
        ]
        
        return jsonify({
            "prediccion_principal": results[0],
            "alternativas": results[1:],
            "texto_analizado": text[:100] + "..." if len(text) > 100 else text
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)