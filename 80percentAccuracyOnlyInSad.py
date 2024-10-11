# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import os
import librosa
import tensorflow as tf
from tqdm import tqdm
from python_speech_features import mfcc
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split  # Import for train-test split

# Import Keras model layers and utilities
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2  # Import for L2 regularization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# Step 2: Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from google.colab import drive
drive.mount('/content/drive')

# Step 3: Copy dataset from Google Drive to Colab's local runtime (for faster access)
!cp -r /content/drive/MyDrive/Audio_DatasetMini100 /content/

# Step 4: Load data from the local directory
root_dir = '/content/Audio_DatasetMini100'  # Path to your data directory

# Step 4: Initialize lists for file paths and labels
paths = []
labels = []

# Loop through each folder (representing an emotion) and each file
for emotion_folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, emotion_folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            paths.append(file_path)
            label = emotion_folder.replace("_emotion", "").lower()
            labels.append(label)

# Create a DataFrame with paths and labels
df = pd.DataFrame({'path': paths, 'label': labels})
df['Length'] = [librosa.get_duration(path=path) for path in df['path']]
print(df.head())

# Model Configurations
class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate / 10)

config = Config(mode='conv')

# Calculate class distribution for balanced training
class_dist = df.groupby(['label'])['Length'].mean()
prop_dist = class_dist / class_dist.sum()
classes = list(np.unique(df.label))

# Parallelized feature extraction function
def extract_features(file_path, label, config):
    signal, rate = librosa.load(file_path, sr=config.rate)
    rand_index = np.random.randint(0, signal.shape[0] - config.step)
    sample = signal[rand_index: rand_index + config.step]
    X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
    return X_sample, label

# Build dataset using parallel processing
def build_rand_feat_parallel(df, n_samples, config, class_dist, prop_dist, classes):
    results = Parallel(n_jobs=-1)(
        delayed(extract_features)(df.loc[np.random.choice(df[df.label == rand_class].index), 'path'],
                                  classes.index(rand_class), config)
        for rand_class in np.random.choice(class_dist.index, size=n_samples, p=prop_dist)
    )
    X, y = zip(*results)
    X, y = np.array(X), np.array(y)

    # Normalizing the features
    _min, _max = np.min(X), np.max(X)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes=len(classes))
    return X, y

# Generate random features using the parallelized function
n_samples = 2 * int(df['Length'].sum() / 0.1)
X, y = build_rand_feat_parallel(df, n_samples, config, class_dist, prop_dist, classes)

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

# Define the CNN model architecture
def get_conv_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
   # model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)))

    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.4))
    model.add(Dense(len(classes), activation='softmax'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Build the CNN model
input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = get_conv_model(input_shape)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

# Train the model using training and validation data
history = model.fit(X_train, y_train,
                    epochs=40,
                    batch_size=64,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(y_true, y_pred_classes, classes)

# Function to plot the misclassification rate for each category
def plot_misclassification_rate(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    misclassification_rates = []

    for i in range(len(classes)):
        correct_predictions = cm[i, i]
        total_predictions = np.sum(cm[i, :])
        misclassification_rate = 1 - (correct_predictions / total_predictions)
        misclassification_rates.append(misclassification_rate)

    # Plotting the misclassification rate for each class
    plt.figure(figsize=(12, 6))
    plt.bar(classes, misclassification_rates)
    plt.xlabel('Classes')
    plt.ylabel('Misclassification Rate')
    plt.title('Misclassification Rate for Each Category')
    plt.show()

# Plot the misclassification rate for each category
plot_misclassification_rate(y_true, y_pred_classes, classes)
