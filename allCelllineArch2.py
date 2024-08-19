# Final - Arch 2
import pandas as pd
import numpy as np


from utils import load_data, preprocess_data, evaluate_model,removeNegativeVals,computeCgContent,generate_kmers,plot_adjusted_charts,plot_combined_charts,balanced_accuracy_score,train_model_with_early_stopping
from models import create_cnn_lstm_model, create_cnn_bilstm_model, create_cnn_gru_model


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

# Check for negative values in the cleavage frequency
# negative_values = data[data['cleavage_freq'] < 0]
# print(f"Number of negative cleavage frequencies: {len(negative_values)}")

# # If negative values exist, drop them
# if len(negative_values) > 0:
#     data = data[data['cleavage_freq'] >= 0]

# # Extract relevant features and compute CG content from the grna_target_sequence
# data['cg_content'] = data['grna_target_sequence'].apply(lambda seq: (seq.count('C') + seq.count('G')) / len(seq))

data=removeNegativeVals(data,'cleavage_freq')
data=computeCgContent(data, 'grna_target_sequence', 'cg_content')

data['kmers'] = data['grna_target_sequence'].apply(generate_kmers)

# Tokenize the k-mer sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['kmers'])
sequences = tokenizer.texts_to_sequences(data['kmers'])
max_len = max(len(seq) for seq in sequences)
kmer_padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Select energy columns and cg_content as features
features = data[['energy_1', 'energy_2', 'energy_3', 'energy_4', 'energy_5', 'cg_content']]
target = data['cleavage_freq']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_train_kmer, X_test_kmer = train_test_split(kmer_padded_sequences, test_size=0.2, random_state=42)

# Binarize the cleavage_freq based on the median as a starting point
optimal_threshold = np.percentile(y_train, 50)
print(f"Optimal Threshold for Cleavage Frequency: {optimal_threshold}")

# Binarize
data['target'] = (data['cleavage_freq'] >= optimal_threshold).astype(int)
y_train_binary = (y_train >= optimal_threshold).astype(int)
y_test_binary = (y_test >= optimal_threshold).astype(int)


X_train = X_train.values
X_test = X_test.values
y_train = y_train_binary.values
y_test = y_test_binary.values


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# best threshold - F1 score
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'CNN+LSTM': create_cnn_lstm_model,
    'CNN+BiDirectional LSTM': create_cnn_bilstm_model,
    'CNN+GRU': create_cnn_gru_model
}

thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
best_thresholds = {}
results = []

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100  

for model_name, create_model in models.items():
    best_f1 = 0
    best_threshold = 0.5

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

            # Evaluate the model on the validation fold using the new evaluate_model function
            metrics, _ = evaluate_model(y_val_fold, y_val_pred_proba, thresholds)
            fold_metrics.append(metrics['f1_score'])

        avg_f1 = np.mean(fold_metrics)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = threshold

    best_thresholds[model_name] = best_threshold

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
        metrics['model'] = model_name
        fold_metrics.append(metrics)

    results.extend(fold_metrics)

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Calculate average metrics
avg_results = results_df.groupby(['model']).mean().reset_index()

print("Average Cross-Validation Metrics for Different Models at Selected Thresholds:")
print(avg_results)

# Prepare data for plotting
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'balanced_accuracy']
metric_values = [avg_results[metric].tolist() for metric in metrics_to_plot]
model_names = avg_results['model'].tolist()

# Plot the metrics using the updated function
plot_adjusted_charts(model_names, metrics_to_plot, metric_values, chart_title='Architecture 2 Model Comparison by ')

# Plotting Precision-Recall Curve
plt.figure(figsize=(12, 8))
for model_name, create_model in models.items():
    # Train the model on the entire training set
    model = create_model(X_train.shape[1], vocab_size, embedding_dim, max_len)
    combined_X_train = np.hstack([X_train, X_train_kmer])
    combined_X_train_resampled, y_train_resampled = smote_tomek.fit_resample(combined_X_train, y_train_binary)

    # Split the resampled data back into features and kmer sequences
    X_train_resampled = combined_X_train_resampled[:, :X_train.shape[1]]
    X_train_kmer_resampled = combined_X_train_resampled[:, X_train.shape[1]:]

    # Ensure the resampled data is 3D for LSTM/GRU
    X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], X_train_resampled.shape[1], 1))
    X_train_kmer_resampled = X_train_kmer_resampled.reshape((X_train_kmer_resampled.shape[0], X_train_kmer_resampled.shape[1]))

    model.fit([X_train_resampled, X_train_kmer_resampled, X_train_resampled[:, :6]], y_train_resampled, epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping])

    # Predict probabilities on the test set
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_test_kmer_reshaped = X_test_kmer.reshape((X_test_kmer.shape[0], X_test_kmer.shape[1]))
    y_test_pred_proba = model.predict([X_test_reshaped, X_test_kmer_reshaped, X_test[:, :6]]).flatten()

    # Calculate precision and recall
    precision, recall, _ = precision_recall_curve(y_test_binary, y_test_pred_proba)
    plt.plot(recall, precision, label=model_name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Different Models')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
