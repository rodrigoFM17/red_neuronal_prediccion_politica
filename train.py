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
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    
    numeric_features = [
        'numero', 'fuentes', 'estadisticas', 'numero adjetivos',
        'terminos ideologicos', 'numero palabras', 'imagenes', 'citas directas'
    ]
    
    available_numeric = [col for col in numeric_features if col in df.columns]
    
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
    
    axes[0].plot(history.history['accuracy'], label='Entrenamiento')
    axes[0].plot(history.history['val_accuracy'], label='Validación')
    axes[0].set_title('Precisión del Modelo')
    axes[0].set_ylabel('Precisión')
    axes[0].set_xlabel('Época')
    axes[0].legend()
    
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

    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
        
        is_hybrid_model = len(model.inputs) > 1
        
        if is_hybrid_model:
            if structured_data is not None and scaler is not None:
                try:
                    if isinstance(structured_data, dict):
                        numeric_values = []
                        
                        for key, value in structured_data.items():
                            if isinstance(value, (int, float)):
                                numeric_values.append(value)
                        
                        structured_array = [numeric_values]
                    else:
                        structured_array = [structured_data]
                    
                    scaled_data = scaler.transform(structured_array)
                    
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
                        
                    prediction = model.predict([padded, full_structured_data])
                except Exception as e:
                    print(f"Error al procesar datos estructurados: {e}")
                    dummy_structured_data = np.zeros((1, model.inputs[1].shape[1]))
                    prediction = model.predict([padded, dummy_structured_data])
            else:
                
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
    
    # 9.5 Configurar la validación cruzada
    
    
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Diccionario para almacenar resultados
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'best_model': None,
        'best_score': 0.0
    }
    
    # 9.6 Ejecutar validación cruzada
    print(f"\nIniciando validación cruzada con {num_folds} folds...")
    
    fold_no = 1
    
    X_text_np = np.array(X_text)
    if X_structured.size > 0:
        X_structured_np = np.array(X_structured)
    else:
        X_structured_np = np.zeros((X_text_np.shape[0], 1)) 
    
    y_np = np.array(y)
    
    for train_idx, test_idx in kf.split(X_text_np):
        print(f"\nEntrenando fold {fold_no}/{num_folds}")
        
        X_train_text, X_test_text = X_text_np[train_idx], X_text_np[test_idx]
        X_train_struct, X_test_struct = X_structured_np[train_idx], X_structured_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]
        
        # 9.7 Construir modelo para este fold
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
        
        # 9.8 Preparar datos de entrada según el tipo de modelo
        if structured_features_dim > 0:
            X_train = [X_train_text, X_train_struct]
            X_test = [X_test_text, X_test_struct]
        else:
            X_train = X_train_text
            X_test = X_test_text
        
        # 9.9 Entrenar modelo para este fold
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,  # Reducir épocas para validación cruzada
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        # 9.10 Evaluar modelo en este fold
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calcular métricas
        acc = accuracy_score(y_test, y_pred_classes)
        prec = precision_score(y_test, y_pred_classes, average='weighted')
        rec = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        
        # Guardar resultados
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1'].append(f1)
        
        print(f"Fold {fold_no} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Si este es el mejor modelo hasta ahora, guardarlo
        if f1 > results['best_score']:
            results['best_score'] = f1
            results['best_model'] = model
        
        fold_no += 1
    
    # 9.11 Mostrar resultados promedio
    print("\nResultados promedio de validación cruzada:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Visualizar resultados de validación cruzada
    plot_cross_validation_results(results)
    
    # 9.12 Evaluar el mejor modelo con matriz de confusión
    best_model = results['best_model']
    
    # Usamos todo el conjunto de datos para generar la matriz de confusión con el mejor modelo
    if structured_features_dim > 0:
        X_full = [X_text_np, X_structured_np]
    else:
        X_full = X_text_np
        
    y_pred = best_model.predict(X_full)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Obtener todas las clases presentes tanto en y como en y_pred_classes
    all_classes = np.unique(np.concatenate([y_np, y_pred_classes]))
    all_labels = label_encoder.inverse_transform(all_classes)
    
    # Matriz de confusión
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_np, y_pred_classes, labels=all_classes)
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_labels, 
                yticklabels=all_labels)
    
    plt.title('Matriz de Confusión (Mejor Modelo)')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 9.13 Guardar el mejor modelo
    best_model = results['best_model']
    best_model.save('modelo_clasificacion_noticias.h5')
    
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
    
    print("Mejor modelo y artefactos guardados correctamente.")
    
    # 9.13 Ejemplo de predicción con el mejor modelo
    ejemplo_texto = """
    El gobierno anuncia nuevas medidas económicas para impulsar el crecimiento 
    y la creación de empleo en las regiones más desfavorecidas. La oposición critica 
    la propuesta calificándola de insuficiente.
    """
    
    try:
        # Determinar cuántas características numéricas espera el modelo
        if structured_features_dim > 0:
            num_features = best_model.inputs[1].shape[1]
            print(f"El modelo espera {num_features} características estructuradas")
            
            # Crear array de ceros como fallback
            dummy_data = np.zeros((1, num_features))
            
            # Realizar predicción con datos dummy
            label, conf, alternatives = predict_political_orientation(
                best_model, tokenizer, label_encoder, ejemplo_texto, dummy_data
            )
        else:
            # Predicción solo con texto
            label, conf, alternatives = predict_political_orientation(
                best_model, tokenizer, label_encoder, ejemplo_texto
            )
        
        if label is not None:
            print(f"\nPredicción para texto de ejemplo:")
            print(f"Clasificación: {label} (confianza: {conf:.2f})")
            print("Alternativas:")
            for alt_label, alt_conf in alternatives:
                print(f"- {alt_label}: {alt_conf:.2f}")
        else:
            print("No se pudo realizar la predicción de ejemplo")
    except Exception as e:
        print(f"Error al hacer predicción de ejemplo: {e}")
        print("No se pudo realizar la predicción de ejemplo")
    
    return best_model, tokenizer, label_encoder, scaler, encoders


# Función auxiliar para visualizar resultados de validación cruzada
def plot_cross_validation_results(results):
    # Preparar datos para gráfico
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]
    
    # Crear gráfico
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=10, alpha=0.7)
    
    # Añadir etiquetas de valores
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{mean:.4f}', ha='center', va='bottom')
    
    plt.title('Resultados de Validación Cruzada')
    plt.ylabel('Puntuación')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Gráfico de cajas para cada métrica
    plt.figure(figsize=(12, 8))
    data_to_plot = [results[m] for m in metrics]
    plt.boxplot(data_to_plot, labels=metrics, patch_artist=True)
    plt.title('Distribución de Métricas en Validación Cruzada')
    plt.ylabel('Puntuación')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return

if __name__ == "__main__":
    # Reemplazar con la ruta de tu dataset
    file_path = "dataset_noticias_politicas.csv"
    main(file_path)