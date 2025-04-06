import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    
    df = pd.read_csv(file_path, delimiter="|")
    
    print(f"Forma del dataset: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")
    print(f"Distribución de clases:\n{df['clasificacion'].value_counts()}")
    
    df['titulo'].fillna('', inplace=True)
    df['contenido'].fillna('', inplace=True)
    
    df['texto_completo'] = df['titulo'] + " " + df['contenido']
    
    return df

def clean_text(text):
    
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'http\S+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text.lower()

def prepare_features(df, max_features=2000):
    
    df['texto_limpio'] = df['texto_completo'].apply(clean_text)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['clasificacion'])
    
    vectorizer = TfidfVectorizer(
        max_features=max_features, 
        min_df=1,           
        max_df=0.8,         
        ngram_range=(1, 2)
    )
    X_text = vectorizer.fit_transform(df['texto_limpio']).toarray()
    
    struct_features = []
    
    numeric_cols = [
        'numero fuentes', 'estadisticas', 'numero adjetivos',
        'terminos ideologicos', 'numero palabras', 'imagenes', 
        'citas directas'
    ]
    
    available_numeric = [col for col in numeric_cols if col in df.columns]
    
    if available_numeric:
        X_numeric = df[available_numeric].copy()
        
        X_numeric.fillna(0, inplace=True)
        
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
        
        struct_features.append(X_numeric_scaled)
    else:
        scaler = None
    
    boolean_cols = ['medio reconocido', 'medio especializado', 'formalidad', 'emocionalidad']
    available_boolean = [col for col in boolean_cols if col in df.columns]
    
    if available_boolean:
        X_boolean = df[available_boolean].copy()
        
        X_boolean.fillna(0, inplace=True)
        
        struct_features.append(X_boolean.values)
    
    if struct_features:
        X_struct = np.hstack(struct_features)
    else:
        X_struct = np.zeros((len(df), 1))  
    
    return X_text, X_struct, y, vectorizer, label_encoder, scaler

def build_neural_network(input_text_dim, input_struct_dim, num_classes):
  
    text_input = Input(shape=(input_text_dim,), name='text_input')
    text_features = Dense(128, activation='relu')(text_input)
    text_features = Dropout(0.5)(text_features)
    text_features = Dense(64, activation='relu')(text_features)
    text_features = Dropout(0.3)(text_features)
    
    if input_struct_dim > 0:
        struct_input = Input(shape=(input_struct_dim,), name='struct_input')
        struct_features = Dense(32, activation='relu')(struct_input)
        struct_features = Dropout(0.3)(struct_features)
        
        combined = Concatenate()([text_features, struct_features])
        inputs = [text_input, struct_input]
    else:
        combined = text_features
        inputs = text_input
    
    x = Dense(32, activation='relu')(combined)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def cross_validate_neural_network(X_text, X_struct, y, n_folds=5, epochs=30, batch_size=16):
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'confusion_matrices': [],
        'best_model': None,
        'best_score': 0.0,
        'best_fold': 0  
    }
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    input_text_dim = X_text.shape[1]
    input_struct_dim = X_struct.shape[1]
    num_classes = len(np.unique(y))
    
    print(f"\nIniciando validación cruzada con {n_folds} folds...")
    
    fold_no = 1
    for train_index, test_index in kf.split(X_text):
        print(f"\nEntrenando fold {fold_no}/{n_folds}")
        
        X_train_text, X_test_text = X_text[train_index], X_text[test_index]
        X_train_struct, X_test_struct = X_struct[train_index], X_struct[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = build_neural_network(input_text_dim, input_struct_dim, num_classes)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        if input_struct_dim > 0:
            X_train = [X_train_text, X_train_struct]
            X_val = [X_test_text, X_test_struct]
        else:
            X_train = X_train_text
            X_val = X_test_text
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        if input_struct_dim > 0:
            y_pred = model.predict([X_test_text, X_test_struct])
        else:
            y_pred = model.predict(X_test_text)
            
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        acc = accuracy_score(y_test, y_pred_classes)
        prec = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred_classes)
        
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1'].append(f1)
        results['confusion_matrices'].append(cm)
        
        print(f"Fold {fold_no} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        if prec > results['best_score']:
            results['best_score'] = prec
            results['best_model'] = model
            results['best_fold'] = fold_no  
        
        fold_no += 1
    
    print("\nResultados promedio de validación cruzada:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\nMejor modelo: Fold {results['best_fold']} con precisión de {results['best_score']:.4f}")
    
    return results

def plot_cross_validation_results(results):
    
    plt.figure(figsize=(12, 6))
    
    n_folds = len(results['precision'])
    fold_numbers = [f"Fold {i+1}" for i in range(n_folds)]
    precision_values = results['precision']
    
    bars = plt.bar(fold_numbers, precision_values, alpha=0.7, color='skyblue')
    
    for bar, val in zip(bars, precision_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{val:.4f}', ha='center', va='bottom')
    
    avg_precision = np.mean(precision_values)
    plt.axhline(y=avg_precision, color='red', linestyle='--', 
                label=f'Promedio: {avg_precision:.4f}')
    
    plt.title('Precisión por Fold en Validación Cruzada')
    plt.ylabel('Precisión')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(results, label_encoder):
   
    confusion_matrices = results['confusion_matrices']
    
    try:
        max_size = max(cm.shape[0] for cm in confusion_matrices)
        
        resized_matrices = []
        for cm in confusion_matrices:
            if cm.shape[0] < max_size:
                new_cm = np.zeros((max_size, max_size), dtype=cm.dtype)
                new_cm[:cm.shape[0], :cm.shape[1]] = cm
                resized_matrices.append(new_cm)
            else:
                resized_matrices.append(cm)
        
        combined_cm = sum(resized_matrices)
        
        labels = label_encoder.classes_[:max_size]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=labels, yticklabels=labels)
        plt.title('Matriz de Confusión Combinada')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"No se pudo generar la matriz de confusión combinada: {e}")

def predict_political_orientation(model, vectorizer, label_encoder, text, X_struct=None):
    
    clean = clean_text(text)
    X_text = vectorizer.transform([clean]).toarray()
    
    if isinstance(model.input, list):
        if X_struct is not None:
            prediction = model.predict([X_text, X_struct])
        else:
            dummy_struct = np.zeros((1, model.input[1].shape[1]))
            prediction = model.predict([X_text, dummy_struct])
    else:
        prediction = model.predict(X_text)
    
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_labels = label_encoder.inverse_transform(top_indices)
    top_probs = prediction[0][top_indices]
    
    predicted_label = top_labels[0]
    confidence = float(top_probs[0])
    
    alternatives = [(label, float(prob)) for label, prob in zip(top_labels, top_probs)]
    
    return predicted_label, confidence, alternatives

def train_final_model(X_text, X_struct, y, cv_results, epochs=30, batch_size=16):
    
    input_text_dim = X_text.shape[1]
    input_struct_dim = X_struct.shape[1]
    num_classes = len(np.unique(y))
    
    
    if 'best_model' in cv_results and cv_results['best_model'] is not None:
        print(f"\nUsando el modelo del mejor fold (Fold {cv_results['best_fold']}) como modelo final")
        final_model = cv_results['best_model']
        
        if input_struct_dim > 0:
            X_combined = [X_text, X_struct]
        else:
            X_combined = X_text
            
        final_model.fit(
            X_combined, y,
            epochs=5,  
            batch_size=batch_size,
            verbose=1
        )
    
    else:
        print("\nEntrenando modelo final con todos los datos")
        final_model = build_neural_network(input_text_dim, input_struct_dim, num_classes)
        final_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        if input_struct_dim > 0:
            X_combined = [X_text, X_struct]
        else:
            X_combined = X_text
        
        history = final_model.fit(
            X_combined, y,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
    
    return final_model


def evaluate_and_plot_final_model(model, X_text, X_struct, y, label_encoder):
    
    if X_struct.shape[1] > 0:
        X_combined = [X_text, X_struct]
    else:
        X_combined = X_text
    
    y_pred = model.predict(X_combined)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    acc = accuracy_score(y, y_pred_classes)
    prec = precision_score(y, y_pred_classes, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred_classes, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y, y_pred_classes)
    
    print("\nMétricas del modelo (evaluado en todos los datos):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    plt.figure(figsize=(14, 12))
    
    labels = label_encoder.classes_
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión del Modelo (todos los datos)')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def main(file_path):
    df = load_and_prepare_data(file_path)
    
    X_text, X_struct, y, vectorizer, label_encoder, scaler = prepare_features(df)
    
    cv_results = cross_validate_neural_network(X_text, X_struct, y)
    
    plot_cross_validation_results(cv_results)
    
    final_model = train_final_model(X_text, X_struct, y, cv_results)
    
    final_metrics = evaluate_and_plot_final_model(final_model, X_text, X_struct, y, label_encoder)
    
    
    final_model.save('modelo_red_neuronal.h5')
    
    with open('vectorizer.pickle', 'wb') as handle:
        pickle.dump(vectorizer, handle)
    
    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle)
        
    if scaler is not None:
        with open('scaler.pickle', 'wb') as handle:
            pickle.dump(scaler, handle)
    
    print("Modelo y artefactos guardados correctamente.")
        
    ejemplo_texto = """
    "La Nueva Era del Progreso: El Gobierno Garantiza un Futuro Brillante para Todos"
Bogotá, 10 de abril de 2025 — Con un firme compromiso hacia el desarrollo y la estabilidad, el gobierno del presidente Martín Guzmán ha lanzado su ambicioso plan "Colombia Avanza", un programa integral que promete transformar el país en una potencia económica y social en la región.

El mandatario anunció que este plan incluye una inversión histórica de 50 billones de pesos en infraestructura, educación y salud, con el objetivo de reducir la desigualdad y garantizar empleo digno para cada colombiano. “Estamos construyendo un país fuerte, próspero y seguro para nuestras familias”, afirmó Guzmán en su discurso ante miles de simpatizantes en la Plaza de Bolívar.

Entre los pilares de este ambicioso programa, destacan:

Educación gratuita y de calidad: El gobierno asegura que ningún joven se quedará sin estudiar. Se destinarán 10 billones de pesos para ampliar el acceso a universidades públicas y mejorar la calidad de la educación.

Crecimiento económico sin precedentes: La economía nacional creció un 7.2% en el último año, superando a todas las economías de la región, gracias a la visión estratégica del presidente Guzmán.

Salud universal: Se implementará un nuevo sistema de salud pública, con hospitales modernos y medicamentos gratuitos para los sectores más vulnerables.

Infraestructura de primer nivel: Se construirán 2,500 kilómetros de carreteras y se modernizarán los sistemas de transporte en las principales ciudades.

“Este es el futuro que todos merecemos”, afirmó el presidente Guzmán. “No permitiremos que la oposición nos detenga con su negativismo. Nuestra nación está avanzando y seguiremos trabajando sin descanso para que cada colombiano viva mejor”.

El gobierno ha iniciado una amplia campaña de difusión en medios de comunicación y redes sociales para informar a la ciudadanía sobre los avances del programa, con el lema “Colombia Avanza, Tú Avanzas”.
    """
    
    
    clean = clean_text(ejemplo_texto)
    X_ejemplo = vectorizer.transform([clean]).toarray()
    
    if X_struct.shape[1] > 0:
        dummy_struct = np.zeros((1, X_struct.shape[1]))
        prediction = final_model.predict([X_ejemplo, dummy_struct])
    else:
        prediction = final_model.predict(X_ejemplo)
    
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_labels = label_encoder.inverse_transform(top_indices)
    top_probs = prediction[0][top_indices]
    
    print(f"\nPredicción para texto de ejemplo:")
    print(f"Clasificación: {top_labels[0]} (confianza: {top_probs[0]:.2f})")
    print("Alternativas:")
    for label, prob in zip(top_labels[1:], top_probs[1:]):
        print(f"- {label}: {prob:.2f}")
    
    return final_model, vectorizer, label_encoder, scaler

if __name__ == "__main__":
    file_path = "dataset_noticias_politicas.csv"
    main(file_path)