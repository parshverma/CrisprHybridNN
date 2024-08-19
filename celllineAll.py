import pandas as pd
import numpy as np


from utils import load_data, preprocess_data, evaluate_model,removeNegativeVals,computeCgContent,generate_kmers,plot_adjusted_charts,plot_combined_charts,balanced_accuracy_score,train_model_with_early_stopping
from models import create_cnn_model, create_lstm_model,create_bilstm_model,create_gru_model,create_cnn_lstm_model, create_cnn_bilstm_model, create_cnn_gru_model,create_cnn_lstm_attention_model, create_cnn_bilstm_attention_model, create_cnn_gru_attention_model

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, precision_recall_curve
from imblearn.combine import SMOTETomek
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Bidirectional, Embedding, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

data=removeNegativeVals(data,'cleavage_freq')
data=computeCgContent(data, 'grna_target_sequence', 'cg_content')

data['kmers'] = data['grna_target_sequence'].apply(generate_kmers)

# Tokenize the k-mer sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['kmers'])
sequences = tokenizer.texts_to_sequences(data['kmers'])
max_len = max(len(seq) for seq in sequences)
kmer_padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define cell lines to process
cell_lines = ['HEK293', 'K562', 'U2OS']

# Initialize the results dictionary if it doesn't already exist
if 'cell_line_results' not in globals():
    cell_line_results = {}

# Process each cell line
for cell_line in cell_lines:
    print(f"Processing for cell line: {cell_line}")

    # Filter data for the current cell line
    cell_line_data = data[data['cell_line'] == cell_line]

    # Tokenize the k-mer sequences for the current cell line
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cell_line_data['kmers'])
    sequences = tokenizer.texts_to_sequences(cell_line_data['kmers'])
    max_len = max(len(seq) for seq in sequences)
    kmer_padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Select energy columns and cg_content as features
    features = cell_line_data[['energy_1', 'energy_2', 'energy_3', 'energy_4', 'energy_5', 'cg_content']]
    target = cell_line_data['cleavage_freq']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train_kmer, X_test_kmer = train_test_split(kmer_padded_sequences, test_size=0.2, random_state=42)

    # Binarize the cleavage_freq based on the median as a starting point
    optimal_threshold = np.percentile(y_train, 50)
    print(f"Optimal Threshold for Cleavage Frequency for {cell_line}: {optimal_threshold}")

    # Binarize the cleavage_freq using the optimal threshold
    cell_line_data['target'] = (cell_line_data['cleavage_freq'] >= optimal_threshold).astype(int)
    y_train_binary = (y_train >= optimal_threshold).astype(int)
    y_test_binary = (y_test >= optimal_threshold).astype(int)

    # Convert DataFrames to NumPy arrays for reshaping
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train_binary.values
    y_test = y_test_binary.values

    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100  # Increased embedding dimension for more nuanced information

    # Define models for each architecture
    models_arch1 = {
        'CNN': create_cnn_model,
        'LSTM': create_lstm_model,
        'BiDirectional LSTM': create_bilstm_model,
        'GRU': create_gru_model
    }

    models_arch2 = {
        'CNN+LSTM': create_cnn_lstm_model,
        'CNN+BiDirectional LSTM': create_cnn_bilstm_model,
        'CNN+GRU': create_cnn_gru_model
    }

    models_arch3 = {
        'CNN+LSTM+Attention': create_cnn_lstm_attention_model,
        'CNN+BiDirectional LSTM+Attention': create_cnn_bilstm_attention_model,
        'CNN+GRU+Attention': create_cnn_gru_attention_model
    }

    # Combine all models to be evaluated
    all_models = {
        'Architecture 1': models_arch1,
        'Architecture 2': models_arch2,
        'Architecture 3': models_arch3
    }

    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

    # Process each architecture
    for arch_name, models in all_models.items():
        best_thresholds = {}
        results = []

        for model_name, create_model in models.items():
            best_f1 = 0
            best_threshold = None

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for threshold in thresholds:
                fold_metrics = []
                for train_index, val_index in kf.split(X_train):
                    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                    X_train_kmer_fold, X_val_kmer_fold = X_train_kmer[train_index], X_train_kmer[val_index]

                    # Apply SMOTE-Tomek to the training fold
                    smote_tomek = SMOTETomek(random_state=42)
                    combined_X_train_fold = np.hstack([X_train_fold, X_train_kmer_fold])
                    combined_X_train_fold_resampled, y_train_fold_resampled = smote_tomek.fit_resample(combined_X_train_fold, y_train_fold)

                    # Split the resampled data back into features and kmer sequences
                    X_train_fold_resampled = combined_X_train_fold_resampled[:, :X_train_fold.shape[1]]
                    X_train_kmer_fold_resampled = combined_X_train_fold_resampled[:, X_train_fold.shape[1]:]

                    # Ensure the resampled data is 3D for LSTM/GRU
                    X_train_fold_resampled = X_train_fold_resampled.reshape((X_train_fold_resampled.shape[0], X_train_fold_resampled.shape[1], 1))
                    X_train_kmer_fold_resampled = X_train_kmer_fold_resampled.reshape((X_train_kmer_fold_resampled.shape[0], X_train_kmer_fold_resampled.shape[1]))

                    # Create and train the model with EarlyStopping
                    model = create_model(X_train_fold_resampled.shape[1], vocab_size, embedding_dim, max_len)
                    model.fit([X_train_fold_resampled, X_train_kmer_fold_resampled, X_train_fold_resampled[:, :6]], y_train_fold_resampled,
                              validation_data=([X_val_fold.reshape((X_val_fold.shape[0], X_val_fold.shape[1], 1)), X_val_kmer_fold, X_val_fold[:, :6]], y_val_fold),
                              epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping])

                    # Predict probabilities
                    X_val_fold_reshaped = X_val_fold.reshape((X_val_fold.shape[0], X_val_fold.shape[1], 1))
                    y_val_pred_proba = model.predict([X_val_fold_reshaped, X_val_kmer_fold, X_val_fold[:, :6]]).flatten()

                    # Evaluate the model on the validation fold to find the best threshold
                    metrics, threshold = evaluate_model(y_val_fold, y_val_pred_proba, [threshold])
                    if metrics and metrics['f1_score'] > best_f1:
                        best_f1 = metrics['f1_score']
                        best_threshold = threshold

            best_thresholds[model_name] = best_threshold if best_threshold is not None else 0.5
            print(f"Best threshold for {model_name} in {cell_line} ({arch_name}): {best_thresholds[model_name]}")

        # Main evaluation with the determined thresholds
        results = []
        for model_name, create_model in models.items():
            best_threshold = best_thresholds[model_name]
            fold_metrics = []
            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                X_train_kmer_fold, X_val_kmer_fold = X_train_kmer[train_index], X_train_kmer[val_index]

                # Apply SMOTE-Tomek to the training fold
                smote_tomek = SMOTETomek(random_state=42)
                combined_X_train_fold = np.hstack([X_train_fold, X_train_kmer_fold])
                combined_X_train_fold_resampled, y_train_fold_resampled = smote_tomek.fit_resample(combined_X_train_fold, y_train_fold)

                # Split the resampled data back into features and kmer sequences
                X_train_fold_resampled = combined_X_train_fold_resampled[:, :X_train_fold.shape[1]]
                X_train_kmer_fold_resampled = combined_X_train_fold_resampled[:, X_train_fold.shape[1]:]

                # Ensure the resampled data is 3D for LSTM/GRU
                X_train_fold_resampled = X_train_fold_resampled.reshape((X_train_fold_resampled.shape[0], X_train_fold_resampled.shape[1], 1))
                X_train_kmer_fold_resampled = X_train_kmer_fold_resampled.reshape((X_train_kmer_fold_resampled.shape[0], X_train_kmer_fold_resampled.shape[1]))

                # Create and train the model with EarlyStopping
                model = create_model(X_train_fold_resampled.shape[1], vocab_size, embedding_dim, max_len)
                model.fit([X_train_fold_resampled, X_train_kmer_fold_resampled, X_train_fold_resampled[:, :6]], y_train_fold_resampled,
                          validation_data=([X_val_fold.reshape((X_val_fold.shape[0], X_val_fold.shape[1], 1)), X_val_kmer_fold, X_val_fold[:, :6]], y_val_fold),
                          epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping])

                # Predict probabilities
                X_val_fold_reshaped = X_val_fold.reshape((X_val_fold.shape[0], X_val_fold.shape[1], 1))
                y_val_pred_proba = model.predict([X_val_fold_reshaped, X_val_kmer_fold, X_val_fold[:, :6]]).flatten()

                # Evaluate the model on the validation fold using the best threshold
                metrics, _ = evaluate_model(y_val_fold, y_val_pred_proba, [best_threshold])
                if metrics:
                    metrics['model'] = model_name
                    fold_metrics.append(metrics)

            results.extend(fold_metrics)

        # Store results for this cell line and architecture
        results_df = pd.DataFrame(results)
        avg_results = results_df.groupby(['model']).mean().reset_index()

        if cell_line in cell_line_results:
            cell_line_results[cell_line][arch_name] = avg_results
        else:
            cell_line_results[cell_line] = {arch_name: avg_results}

        print(f"Average Cross-Validation Metrics for {cell_line} ({arch_name}):")
        print(avg_results)

plot_combined_charts(cell_line_results)
