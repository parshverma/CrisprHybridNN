import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, precision_recall_curve, roc_curve
from imblearn.combine import SMOTETomek
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Bidirectional, Embedding, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


file_path = 'base2_motif.csv'
data = pd.read_csv(file_path)


def removeNegativeVals(data, column):
    
    # Check for negative values in the specified column
    negative_values = data[data[column] < 0]
    print(f"Number of negative values in '{column}': {len(negative_values)}")

    # If negative values exist, drop them
    if len(negative_values) > 0:
        data = data[data[column] >= 0]
    
    return data

def computeCgContent(data, sequence_column, output_column='cg_content'):
    
    # Compute CG content and add it to the specified output column
    data[output_column] = data[sequence_column].apply(
        lambda seq: (seq.count('C') + seq.count('G')) / len(seq)
    )
    
    return data

# k-mer sequences (k=2) code
def generate_kmers(sequence, k=2):
    return ' '.join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


# Evaluate model performance and return the best threshold
def evaluate_model(y_true, y_pred_proba, thresholds):
    best_threshold = None
    best_f1 = 0
    best_metrics = None

    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred_threshold)
        precision = precision_score(y_true, y_pred_threshold, zero_division=0)
        recall = recall_score(y_true, y_pred_threshold)
        f1 = f1_score(y_true, y_pred_threshold)
        auc = roc_auc_score(y_true, y_pred_proba)
        loss = np.mean((y_pred_proba - y_true.reshape(-1, 1)) ** 2)
        balanced_acc = balanced_accuracy_score(y_true, y_pred_threshold)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'balanced_accuracy': balanced_acc,
                'optimal_threshold': threshold
            }

    return best_metrics, best_threshold


# Plot charts based on the provided metrics and values
def plot_adjusted_charts(models, metrics, values, chart_title):
    num_metrics = len(metrics)
    num_models = len(models)

    # color map
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(models, values[i], color=colors[:num_models])
        ax.set_title(f'{chart_title} by - {metric}')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.lower())

        # Set individual y-axis limits with a small margin
        min_value = min(values[i])
        max_value = max(values[i])
        range_margin = (max_value - min_value) * .9  
        ax.set_ylim(min_value - range_margin, max_value + range_margin)

        # text labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + range_margin * 0.02,
                    f'{height:.4f}', ha='center', va='bottom')

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(num_models)]
    fig.legend(handles, models, loc='upper center', ncol=num_models, frameon=False, fontsize='large')

    # For spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_combined_charts(cell_line_results):
    architectures = ['Architecture 1', 'Architecture 2', 'Architecture 3']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'balanced_accuracy']

   
    colors = ['#9467bd', '#17becf', '#e377c2']

 
    for cell_line, arch_results in cell_line_results.items():
        models = []
        values_by_architecture = []

        for arch in architectures:
            arch_data = arch_results[arch]
            models.append(arch_data['model'].tolist())
            values_by_architecture.append([arch_data[metric].tolist() for metric in metrics])

        num_metrics = len(metrics)
        num_architectures = len(architectures)

        flattened_models = []
        short_model_names = []
        identifier = 1
        # track of model names with identifiers
        model_identifiers = []  
        for arch_idx, (arch, model_list) in enumerate(zip(architectures, models)):
            for model in model_list:
                flattened_models.append(f'{model}')
                # Assign an identifier for the model
                model_identifier = f'M{identifier}'
                short_model_names.append(model_identifier)
                model_identifiers.append(f'{model_identifier}: {model} ({arch})')
                identifier += 1

        flattened_values = []
        for metric_values in zip(*values_by_architecture):
            combined_values = []
            for arch_values in metric_values:
                combined_values.extend(arch_values)
            flattened_values.append(combined_values)

       
        if len(flattened_models) != len(flattened_values[0]):
            raise ValueError("Mismatch between the number of models and the number of values provided.")

        fig, axes = plt.subplots(2, 3, figsize=(30, 16)) 
        axes = axes.flatten()

       
        bar_width = 0.6 
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bar_positions = np.arange(len(flattened_models))

           
            current_position = 0
            for j, arch in enumerate(architectures):
                num_models_in_arch = len(models[j])
                ax.bar(bar_positions[current_position:current_position + num_models_in_arch],
                       flattened_values[i][current_position:current_position + num_models_in_arch],
                       width=bar_width, color=colors[j], label=arch)
                current_position += num_models_in_arch

            ax.set_title(f'{cell_line} Model Comparison by {metric.capitalize()}', fontsize=20)
            ax.set_xlabel('Model', fontsize=18)
            ax.set_ylabel(metric.lower(), fontsize=18)

            
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(short_model_names, rotation=0, ha='center', fontsize=16)

          
            min_value = min(flattened_values[i])
            max_value = max(flattened_values[i])
            range_margin = (max_value - min_value) * 0.4  # 40% margin for better spacing
            ax.set_ylim(min_value - range_margin, max_value + range_margin)

            # Adding text labels
            for j, bar_position in enumerate(bar_positions):
                height = flattened_values[i][j]
                ax.text(bar_position, height + range_margin * 0.02,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=16)

        # Adding legend for architectures only
        handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(num_architectures)]
        fig.legend(handles, architectures, loc='upper center', ncol=num_architectures, frameon=False, fontsize=22)

        # Adding another legend for model identifiers without colors, moved to the bottom
        fig.text(0.5, -0.1, "\n".join(model_identifiers), ha='center', fontsize=20)

        # Adjust layout to ensure plots are well spaced and allow room for the bottom legend
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to use the blank space
        plt.show()

# After running the neural network code for all architectures
plot_combined_charts(cell_line_results)


# code for train the model with EarlyStopping
def train_model_with_early_stopping(model, X_train, X_train_kmer, y_train, X_val, X_val_kmer, y_val):
    history = model.fit(
        [X_train, X_train_kmer, X_train[:, :6]], y_train,
        validation_data=([X_val, X_val_kmer, X_val[:, :6]], y_val),
        epochs=100,  # You can set this to a high value since early stopping will handle the stopping
        batch_size=32,
        verbose=0,
        callbacks=[early_stopping]
    )
    return history
