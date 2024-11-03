from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the corrected labels in the intended order
corrected_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

def plot_corrected_confusion_matrix(X, Y, title):
    # Get model predictions
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Apply the specific column swaps as per instructions:
    # 6 -> 3 (Unknown becomes Happy)
    # 5 -> 4 (Sad becomes Neutral)
    # 4 -> 5 (Neutral becomes Sad)
    corrected_predicted_classes = np.copy(predicted_classes)
    
    
   # corrected_predicted_classes[predicted_classes == 4] = 3  # Change "Neutral" (4) to "Sad" (5)
   # corrected_predicted_classes[predicted_classes == 5] = 4  # Change "Sad" (5) to "Neutral" (4)
   # corrected_predicted_classes[predicted_classes == 6] = 5  # Change "Unknown" (6) to "Happy" (3)
    # Generate the confusion matrix with the correctly aligned labels
    conf_matrix_corrected = confusion_matrix(Y, corrected_predicted_classes, labels=range(6))
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix_corrected, annot=True, fmt='d', cmap='Blues',
                xticklabels=corrected_labels, yticklabels=corrected_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {title} (6 emotions)')
    plt.show()
    
    # Print classification report with the corrected column alignment
    print(f"\nClassification Report for {title} (6 emotions):\n")
    print(classification_report(Y, corrected_predicted_classes, target_names=corrected_labels, zero_division=0))

# Use the function to generate metrics and the confusion matrix for OAF and YAF subsets
plot_corrected_confusion_matrix(X_OAF, Y_OAF, 'OAF Subset')
plot_corrected_confusion_matrix(X_YAF, Y_YAF, 'YAF Subset')
