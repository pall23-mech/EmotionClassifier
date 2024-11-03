import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization, LSTM, Activation, Flatten, Multiply
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Mount Google Drive if needed
from google.colab import drive
drive.mount('/content/drive')

# Paths
dataset_path = '/content/drive/MyDrive/Audio_Dataset500'
features_csv_path = '/content/drive/MyDrive/audio_features.csv'

# Loading dataset paths and labels
paths, labels = [], []
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        paths.append(file_path)
        labels.append(filename.split('_')[-1].split('.')[0].lower())

df = pd.DataFrame({'path': paths, 'label': labels})
print(df.head())

# Data Augmentation Functions
def noise(data):
    noise_amp = 0.04 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.70):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.8):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def higher_speed(data, speed_factor=1.25):
    return librosa.effects.time_stretch(data, rate=speed_factor)

def lower_speed(data, speed_factor=0.75):
    return librosa.effects.time_stretch(data, rate=speed_factor)

# Feature Extraction
def extract_features(data, sample_rate=22050, n_mfcc=58):
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def get_features(path):
    data, sample_rate = librosa.load(path, duration=3, offset=0.5)
    features = [extract_features(data, sample_rate=sample_rate)]

    for aug_func in [noise, stretch, shift, lambda x: pitch(x, sample_rate), higher_speed, lower_speed]:
        try:
            aug_data = aug_func(data)
            features.append(extract_features(aug_data, sample_rate=sample_rate))
        except Exception as e:
            print(f"Error in augmentation: {e}")

    return features

# Load or Extract Features
try:
    features_df = pd.read_csv(features_csv_path)
except FileNotFoundError:
    X, Y = [], []
    for path, label in zip(paths, labels):
        features = get_features(path)
        for feat in features:
            X.append(feat)
            Y.append(label)
    features_df = pd.DataFrame(X)
    features_df['label'] = Y
    features_df.to_csv(features_csv_path, index=False)

# Compute training mean and std from extracted features
scaler = StandardScaler()
X_features = features_df.drop('label', axis=1).values
scaler.fit(X_features)
TRAINING_MEAN = scaler.mean_
TRAINING_STD = np.sqrt(scaler.var_)
print("TRAINING_MEAN:", TRAINING_MEAN)
print("TRAINING_STD:", TRAINING_STD)

# Register SelfAttention layer with Keras
@tf.keras.utils.register_keras_serializable()
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.dense = Dense(units, activation='softmax')

    def call(self, inputs):
        attention = self.dense(inputs)
        attention_output = Multiply()([inputs, attention])
        return attention_output

# Split Data
X = scaler.transform(features_df.drop('label', axis=1))  # Apply scaling here
Y = features_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# One-Hot Encode
encoder = OneHotEncoder()
y_train = encoder.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_test = encoder.transform(np.array(y_test).reshape(-1, 1)).toarray()

# Expand Dimensions for Conv1D
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Model Creation (CNN-LSTM with Self-Attention)
model = Sequential()
input_shape = (X_train.shape[1], X_train.shape[2])

# Additional Conv1D layers for deeper feature extraction
model.add(Conv1D(64, 3, input_shape=input_shape, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(128, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))

# LSTM layers with more units
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.15))

# Self-attention layer
model.add(SelfAttention(units=128))  # Add attention layer

# Flatten and Dense layers for classification
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# Training with ReduceLROnPlateau
rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, min_lr=1e-6)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[rlrp])

# Plot Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Model Evaluation
train_loss, train_accuracy = model.evaluate(X_train, y_train)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Testing Accuracy: {test_accuracy:.2%}")

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Save Model
model_save_path = '/content/drive/MyDrive/my_model_newSelfAttention.keras'
model.save(model_save_path)
