import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

# Mount Google Drive if needed
from google.colab import drive
drive.mount('/content/drive')

# Define the path to your dataset and model
dataset_path = '/content/drive/MyDrive/Toronto'
model_path = '/content/drive/MyDrive/full_model.keras'

# Define and register the SelfAttention layer for serialization
@tf.keras.utils.register_keras_serializable()
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units
        self.dense = tf.keras.layers.Dense(units, activation='softmax')

    def call(self, inputs):
        attention = self.dense(inputs)
        return tf.keras.layers.multiply([inputs, attention])

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load the model with the custom SelfAttention layer
custom_objects = {'SelfAttention': SelfAttention}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
print("Model loaded successfully.")

# Define the expected labels explicitly, excluding "pleasant"
expected_labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
label_to_int = {label: idx for idx, label in enumerate(expected_labels)}

# Define helper functions
def extract_label_from_path(file_path):
    """Extract label and prefix from the folder name."""
    folder_name = os.path.basename(os.path.dirname(file_path))
    label = folder_name.split('_')[1].lower()  # Normalize label to lowercase
    prefix = folder_name.split('_')[0]  # Extract OAF or YAF prefix
    return label, prefix

def extract_features(data, sample_rate=22050, n_mfcc=58):
    """Extract MFCC features from audio data."""
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def get_features_and_label(path):
    """Load audio file, extract features, and label."""
    label, prefix = extract_label_from_path(path)
    if label not in expected_labels:  # Exclude labels not in expected labels
        return None, None, None
    try:
        data, sample_rate = librosa.load(path, duration=3, offset=0.5)
        features = extract_features(data, sample_rate=sample_rate)
        return features, label_to_int[label], prefix
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None, None, None

# Load all audio file paths and extract their labels
audio_paths = []
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            audio_paths.append(os.path.join(root, file))

# Extract features and labels in parallel
X_new, Y_new, prefixes = [], [], []
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(get_features_and_label, audio_paths))

# Filter out None values and separate features, labels, and prefixes
for features, label, prefix in results:
    if features is not None and label is not None and prefix is not None:
        X_new.append(features)
        Y_new.append(label)
        prefixes.append(prefix)

# Convert lists to arrays and preprocess
X_new = np.array(X_new)
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)
X_new = np.expand_dims(X_new, axis=2)  # Expand dimensions for Conv1D

# Convert Y_new and prefixes to numpy arrays for indexing
Y_new = np.array(Y_new)
prefixes = np.array(prefixes)

# Separate data for OAF and YAF subsets
X_OAF = X_new[prefixes == 'OAF']
Y_OAF = Y_new[prefixes == 'OAF']
X_YAF = X_new[prefixes == 'YAF']
Y_YAF = Y_new[prefixes == 'YAF']

# Function to generate predictions and plot confusion matrix
def plot_confusion_matrix(X, Y, title):
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)

    # Generate the confusion matrix only for valid labels
    conf_matrix = confusion_matrix(Y, predicted_classes, labels=range(len(expected_labels)))

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=expected_labels, yticklabels=expected_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()

    # Print classification report
    print(f"\nClassification Report for {title}:\n")
    print(classification_report(Y, predicted_classes, target_names=expected_labels))

# Plot confusion matrices for OAF and YAF subsets
plot_confusion_matrix(X_OAF, Y_OAF, 'OAF Subset')
plot_confusion_matrix(X_YAF, Y_YAF, 'YAF Subset')
