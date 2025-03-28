import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import pickle
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
    df['pais de origen'].fillna('desconocido', inplace=True)
    df['autor'].fillna('desconocido', inplace=True)
    df['tema principal'].fillna('desconocido', inplace=True)
    
    # Columnas numéricas - rellenar con 0
    numeric_cols = ['numero', 'fuentes', 'estadisticas', 'numero adjetivos', 
                    'terminos ideologicos', 'numero palabras', 'imagenes', 'citas directas']
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    # Columnas booleanas - rellenar con 0
    bool_cols = ['medio reconocido', 'medio especializado', 'formalidad', 'emocionalidad']
    for col in bool_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    # Combinar título y contenido para tener más contexto
    df['texto_completo'] = df['titulo'] + " " + df['contenido']
    
    return df

# 2. Preprocesamiento del texto
def preprocess_text(df, max_words=10000, max_length=200):
    """
    Tokeniza y vectoriza el texto para la red neuronal
    """
    # Inicializar tokenizador
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['texto_completo'])
    
    # Convertir textos a secuencias
    sequences = tokenizer.texts_to_sequences(df['texto_completo'])
    
    # Padding de secuencias
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['clasificacion'])
    
    return padded_sequences, encoded_labels, tokenizer, label_encoder

# 3. Preparación de características estructuradas
def prepare_structured_features(df):
    """
    Prepara las características estructuradas del dataset
    """
    # 3.1 Características numéricas
    numeric_features = [
        'numero', 'fuentes', 'estadisticas', 'numero adjetivos',
        'terminos ideologicos', 'numero palabras', 'imagenes', 'citas directas'
    ]
    
    # Verificar cuáles columnas están disponibles
    available_numeric = [col for col in numeric_features if col in df.columns]
    
    # Crear matriz de características numéricas
    if available_numeric:
        numeric_data = df[available_numeric].values
        
        # Escalar características numéricas
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_data)
    else:
        numeric_scaled = np.array([]).reshape(df.shape[0], 0)
        scaler = None
    
    # 3.2 Características categóricas
    # 3.2.1 Características booleanas
    boolean_features = ['medio reconocido', 'medio especializado', 'formalidad', 'emocionalidad']
    available_boolean = [col for col in boolean_features if col in df.columns]
    
    if available_boolean:
        boolean_data = df[available_boolean].values
    else:
        boolean_data = np.array([]).reshape(df.shape[0], 0)
    
    # 3.2.2 Características categóricas con one-hot encoding
    categorical_features = ['pais de origen', 'tema principal']
    available_categorical = [col for col in categorical_features if col in df.columns]
    
    if available_categorical:
        # Inicializar OneHotEncoder
        encoders = {}
        encoded_data_list = []
        
        for feature in available_categorical:

            try:
                encoders[feature] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                # Para versiones anteriores de scikit-learn
                encoders[feature] = OneHotEncoder(sparse=False, handle_unknown='ignore')
            # Reshape para que sea 2D como requiere el OneHotEncoder
            feature_encoded = encoders[feature].fit_transform(df[[feature]])
            encoded_data_list.append(feature_encoded)
        
        # Combinar todas las características one-hot encoded si hay alguna
        if encoded_data_list:
            categorical_encoded = np.hstack(encoded_data_list)
        else:
            categorical_encoded = np.array([]).reshape(df.shape[0], 0)
    else:
        categorical_encoded = np.array([]).reshape(df.shape[0], 0)
        encoders = {}
    
    # 3.3 Combinar todas las características estructuradas
    structured_features = np.hstack([
        numeric_scaled if numeric_scaled.size > 0 else np.zeros((df.shape[0], 0)),
        boolean_data if boolean_data.size > 0 else np.zeros((df.shape[0], 0)),
        categorical_encoded if categorical_encoded.size > 0 else np.zeros((df.shape[0], 0))
    ])
    
    return structured_features, scaler, encoders

# 4. Modelo híbrido
def build_hybrid_model(vocab_size, embedding_dim=128, max_length=200, 
                       structured_features_dim=0, num_classes=13):
    """
    Construye un modelo híbrido que combina texto y características estructuradas
    """
    # 4.1 Rama de procesamiento de texto
    text_input = Input(shape=(max_length,), name='text_input')
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(text_input)
    
    # Capas LSTM bidireccionales
    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    dropout1 = Dropout(0.2)(lstm1)
    
    lstm2 = Bidirectional(LSTM(32))(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    
    # 4.2 Rama de características estructuradas
    if structured_features_dim > 0:
        structured_input = Input(shape=(structured_features_dim,), name='structured_input')
        structured_dense = Dense(32, activation='relu')(structured_input)
        structured_dropout = Dropout(0.2)(structured_dense)
        
        # 4.3 Concatenar ambas ramas
        concatenated = Concatenate()([dropout2, structured_dropout])
    else:
        structured_input = None
        concatenated = dropout2
    
    # 4.4 Capas finales
    dense = Dense(64, activation='relu')(concatenated)
    dropout3 = Dropout(0.5)(dense)
    output = Dense(num_classes, activation='softmax')(dropout3)
    
    # 4.5 Definir modelo
    if structured_features_dim > 0:
        model = Model(inputs=[text_input, structured_input], outputs=output)
    else:
        model = Model(inputs=text_input, outputs=output)
    
    # Compilar modelo
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# 5. Entrenamiento y evaluación
def train_and_evaluate(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    """
    Entrena el modelo y evalúa su rendimiento
    """
    # Early stopping para evitar overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluar modelo
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Pérdida de validación: {loss}")
    print(f"Precisión de validación: {accuracy}")
    
    return model, history

# 6. Visualización de métricas
def plot_metrics(history):
    """
    Visualiza las métricas de entrenamiento
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de precisión
    axes[0].plot(history.history['accuracy'], label='Entrenamiento')
    axes[0].plot(history.history['val_accuracy'], label='Validación')
    axes[0].set_title('Precisión del Modelo')
    axes[0].set_ylabel('Precisión')
    axes[0].set_xlabel('Época')
    axes[0].legend()
    
    # Gráfico de pérdida
    axes[1].plot(history.history['loss'], label='Entrenamiento')
    axes[1].plot(history.history['val_loss'], label='Validación')
    axes[1].set_title('Pérdida del Modelo')
    axes[1].set_ylabel('Pérdida')
    axes[1].set_xlabel('Época')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# 7. Análisis de resultados
def analyze_results(model, X_test, y_test, label_encoder):
    """
    Analiza y visualiza los resultados del modelo
    """
    # Predicciones
    if isinstance(X_test, list):  # Para modelo híbrido
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Obtener todas las clases presentes tanto en y_test como en y_pred_classes
    all_classes = np.unique(np.concatenate([y_test, y_pred_classes]))
    all_labels = label_encoder.inverse_transform(all_classes)
    
    # Crear diccionario para mapear índices numéricos a nombres de clases
    labels_dict = {i: label for i, label in zip(all_classes, all_labels)}
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred_classes, labels=all_classes, 
                               target_names=all_labels))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_classes, labels=all_classes)
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_labels, 
                yticklabels=all_labels)
    
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# 8. Función de predicción
def predict_political_orientation(model, tokenizer, label_encoder, 
                               text, structured_data=None, 
                               scaler=None, encoders=None,
                               max_length=200):
    """
    Predice la orientación política de un nuevo texto
    """
    try:
        # Preprocesar texto
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
        
        # Determinar si es un modelo híbrido verificando sus inputs
        is_hybrid_model = len(model.inputs) > 1
        
        # Preparar datos de entrada
        if is_hybrid_model:
            if structured_data is not None and scaler is not None:
                try:
                    # Convertir structured_data de diccionario a lista de valores
                    if isinstance(structured_data, dict):
                        # Extraer solo valores numéricos y ordenarlos según las columnas que el scaler conoce
                        numeric_values = []
                        
                        # Intentar extraer solo valores numéricos
                        for key, value in structured_data.items():
                            if isinstance(value, (int, float)):
                                numeric_values.append(value)
                        
                        structured_array = [numeric_values]
                    else:
                        structured_array = [structured_data]
                    
                    # Preparar características estructuradas si están disponibles
                    scaled_data = scaler.transform(structured_array)
                    
                    # Añadir one-hot encoding si hay encoders
                    if encoders:
                        encoded_features = []
                        for feature_name, encoder in encoders.items():
                            if isinstance(structured_data, dict) and feature_name in structured_data:
                                feature_value = structured_data[feature_name]
                                try:
                                    encoded = encoder.transform([[feature_value]])
                                    encoded_features.append(encoded)
                                except:
                                    pass
                        
                        if encoded_features:
                            encoded_data = np.hstack(encoded_features)
                            full_structured_data = np.hstack([scaled_data, encoded_data])
                        else:
                            full_structured_data = scaled_data
                    else:
                        full_structured_data = scaled_data
                        
                    # Realizar predicción para modelo híbrido
                    prediction = model.predict([padded, full_structured_data])
                except Exception as e:
                    print(f"Error al procesar datos estructurados: {e}")
                    # Si hay algún error en el procesamiento de datos estructurados, usar zeros
                    dummy_structured_data = np.zeros((1, model.inputs[1].shape[1]))
                    prediction = model.predict([padded, dummy_structured_data])
            else:
                # Si no tenemos datos estructurados pero el modelo es híbrido,
                # creamos datos vacíos con ceros
                dummy_structured_data = np.zeros((1, model.inputs[1].shape[1]))
                prediction = model.predict([padded, dummy_structured_data])
        else:
            # Realizar predicción solo con texto
            prediction = model.predict(padded)
        
        # Obtener clase predicha
        predicted_class = np.argmax(prediction[0])
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        confidence = float(prediction[0][predicted_class])
        
        # Obtener las 3 categorías más probables
        top_k_indices = np.argsort(prediction[0])[-3:][::-1]
        top_k_labels = label_encoder.inverse_transform(top_k_indices)
        top_k_probs = prediction[0][top_k_indices]
        
        alternative_predictions = [(label, float(prob)) for label, prob in zip(top_k_labels, top_k_probs)]
        
        return predicted_label, confidence, alternative_predictions
    
    except Exception as e:
        print(f"Error en predict_political_orientation: {e}")
        return None, None, None

# 9. Función principal
def main(file_path):
    # 9.1 Cargar y preparar datos
    df = load_and_prepare_data(file_path)
    
    # 9.2 Preprocesar texto
    X_text, y, tokenizer, label_encoder = preprocess_text(df)
    
    # 9.3 Preparar características estructuradas
    X_structured, scaler, encoders = prepare_structured_features(df)
    
    # 9.4 Verificar si necesitamos sobremuestrear para clases minoritarias
    # Contar ejemplos por clase
    class_counts = np.bincount(y)
    min_samples = 3  # Mínimo número de ejemplos por clase para sobremuestrear
    
    if np.any(class_counts < min_samples):
        print(f"Se detectaron clases con menos de {min_samples} ejemplos.")
        print("Distribución original:", class_counts)
        
        # Combinar características textuales y estructuradas para SMOTE
        X_combined = np.hstack([X_text, X_structured])
        
        try:
            # Intentar aplicar SMOTE
            smote = SMOTE(random_state=42, k_neighbors=1)
            X_resampled, y_resampled = smote.fit_resample(X_combined, y)
            
            # Separar de nuevo las características
            X_text = X_resampled[:, :X_text.shape[1]]
            X_structured = X_resampled[:, X_text.shape[1]:]
            y = y_resampled
            
            print("Distribución después de SMOTE:", np.bincount(y))
        except Exception as e:
            print(f"No se pudo aplicar SMOTE: {e}")
            print("Continuando con los datos originales...")
    
    # 9.5 Dividir en conjuntos de entrenamiento, validación y prueba
    # Primero ver si tenemos suficientes ejemplos para stratify
    class_counts = np.bincount(y)
    if np.any(class_counts < 2):
        # No usar stratify si hay clases con menos de 2 ejemplos
        X_train_text, X_temp_text, X_train_structured, X_temp_structured, y_train, y_temp = train_test_split(
            X_text, X_structured, y, test_size=0.3, random_state=42
        )
        
        X_val_text, X_test_text, X_val_structured, X_test_structured, y_val, y_test = train_test_split(
            X_temp_text, X_temp_structured, y_temp, test_size=0.5, random_state=42
        )
    else:
        # Usar stratify si todas las clases tienen al menos 2 ejemplos
        X_train_text, X_temp_text, X_train_structured, X_temp_structured, y_train, y_temp = train_test_split(
            X_text, X_structured, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_val_text, X_test_text, X_val_structured, X_test_structured, y_val, y_test = train_test_split(
            X_temp_text, X_temp_structured, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    
    print(f"Tamaño del conjunto de entrenamiento: {len(X_train_text)}")
    print(f"Tamaño del conjunto de validación: {len(X_val_text)}")
    print(f"Tamaño del conjunto de prueba: {len(X_test_text)}")
    
    # 9.6 Construir modelo
    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(np.unique(y))
    structured_features_dim = X_structured.shape[1] if X_structured.size > 0 else 0
    
    model = build_hybrid_model(
        vocab_size, 
        embedding_dim=128, 
        max_length=X_text.shape[1], 
        structured_features_dim=structured_features_dim,
        num_classes=num_classes
    )
    
    model.summary()
    
    # 9.7 Entrenar y evaluar modelo
    # Preparar datos de entrada según el tipo de modelo
    if structured_features_dim > 0:
        X_train = [X_train_text, X_train_structured]
        X_val = [X_val_text, X_val_structured]
        X_test = [X_test_text, X_test_structured]
    else:
        X_train = X_train_text
        X_val = X_val_text
        X_test = X_test_text
    
    trained_model, history = train_and_evaluate(
        model, X_train, y_train, X_val, y_val, epochs=15
    )
    
    # 9.8 Visualizar métricas
    plot_metrics(history)
    
    # 9.9 Analizar resultados
    analyze_results(trained_model, X_test, y_test, label_encoder)
    
    # 9.10 Guardar modelo y artefactos
    trained_model.save('modelo_clasificacion_noticias.h5')
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if scaler is not None:
        with open('scaler.pickle', 'wb') as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if encoders:
        with open('encoders.pickle', 'wb') as handle:
            pickle.dump(encoders, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Modelo y artefactos guardados correctamente.")
    
    # 9.11 Ejemplo de predicción
    ejemplo_texto = """
    El miércoles ha arrancado con una novedad importante: la jueza ha citado a la pareja de Ayuso como investigado por corrupción en los negocios. Alberto González Amador deberá declarar como imputado por corrupción en los negocios y administración desleal el próximo 10 de abril ante la jueza que ya le investigaba por fraude fiscal.

De por sí, la jornada ya venía cargada. El presidente del Gobierno, Pedro Sánchez, y el líder del PP, Alberto Núñez Feijóo, han vuelto a enfrentarse en un cara a cara en la sesión de control del Congreso, en un ambiente político recalentado por el pacto del PP con Vox en la Comunitat Valenciana que mantiene a Carlos Mazón al frente de la Generalitat, y por el decreto de reparto de menores migrantes en las comunidades autónomas. Sobre esto, te contamos:El PP frena los pactos presupuestarios en otras comunidades tras el acuerdo de Mazón mientras Vox mete presión. Cuca Gamarra asegura que el acuerdo valenciano se debe a la “situación excepcional” provocada por la DANA, mientras el partido de Abascal ve “evidente” que en la Comunitat Valenciana han asumido su discurso
El Gobierno pasa a la ofensiva institucional contra Mazón con las políticas climáticas y de inmigración. El Ejecutivo desbloquea el reparto autonómico de menores migrantes un día después de que el presidente valenciano se comprometiera a “no admitir más repartos” y promete permanecer “vigilante” ante el posible incumplimiento del Pacto Verde europeo
Además, este miércoles la Audiencia Provincial de Madrid ha dado a conocer la sentencia del juicio a los empresarios Alberto Luceño y Luis Medina por el 'caso Mascarillas', una compraventa de material sanitario con el Ayuntamiento de Madrid en plena pandemia por la que cobraron seis millones de euros. Ambos han resultado absueltos del delito de estafa, pero Luceño ha sido condenado por otros dos delitos.
    """
    
    # Ejemplo de datos estructurados
    ejemplo_estructurado = {
        'numero': 1,
        'fuentes': 0,
        'estadisticas': 2,
        'numero_adjetivos': 5,
        'terminos_ideologicos': 6,
        'numero_palabras': 276,
        'imagenes': 1,
        'citas_directas': 2,
        'medio_reconocido': 0,
        'medio_especializado': 0,
        'formalidad': 1,
        'emocionalidad': 0,
        'pais_de_origen': 'España',
        'tema_principal': 'Justicia y Estado de Derecho'
    }
    
    try:
        label, conf, alternatives = predict_political_orientation(
            trained_model, tokenizer, label_encoder, 
            ejemplo_texto, ejemplo_estructurado,
            scaler, encoders
        )
        
        print(f"\nPredicción para texto de ejemplo:")
        print(f"Clasificación: {label} (confianza: {conf:.2f})")
        print("Alternativas:")
        for alt_label, alt_conf in alternatives:
            print(f"- {alt_label}: {alt_conf:.2f}")
    except Exception as e:
        print(f"Error al hacer predicción de ejemplo: {e}")
        print("Realizando predicción solo con texto...")
        
        label, conf, alternatives = predict_political_orientation(
            trained_model, tokenizer, label_encoder, ejemplo_texto
        )
        
        print(f"\nPredicción para texto de ejemplo (solo texto):")
        print(f"Clasificación: {label} (confianza: {conf:.2f})")
        print("Alternativas:")
        for alt_label, alt_conf in alternatives:
            print(f"- {alt_label}: {alt_conf:.2f}")
    
    return trained_model, tokenizer, label_encoder, scaler, encoders

if __name__ == "__main__":
    # Reemplazar con la ruta de tu dataset
    file_path = "dataset_noticias_politicas.csv"
    main(file_path)