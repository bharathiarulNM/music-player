import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import librosa.display
warnings.filterwarnings("ignore")

# Define dataset path
DATA_DIR = 'music_dataset'  # adjust if needed
LABELS = ['down', 'up']

# Data containers
X = []
y = []

# Function to extract and clean audio features
def extract_mfcc(filepath, max_pad_len=130):
    try:
        # Load audio
        audio, sr = librosa.load(filepath, sr=16000)  # Resample to 16kHz
        # Trim silence
        audio, _ = librosa.effects.trim(audio)
        # Normalize audio
        audio = librosa.util.normalize(audio)
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        # Pad MFCC to consistent shape
        mfcc = pad_sequences([mfcc.T], maxlen=max_pad_len, padding='post', dtype='float32')
        return mfcc[0]
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# Loop through dataset and process audio files
for label in LABELS:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            file_path = os.path.join(folder, file)
            features = extract_mfcc(file_path)
            if features is not None:
                X.append(features)
                y.append(0 if label == 'down' else 1)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset shape
print("✅ Preprocessing done.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Visualization Functions

# 1. Plot waveform
def plot_waveform(label):
    path = os.path.join(DATA_DIR, label)
    file = os.listdir(path)[0]  # pick the first file
    file_path = os.path.join(path, file)
    
    signal, sr = librosa.load(file_path, sr=16000)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(signal, sr=sr)
    plt.title(f"Waveform - {label}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

# 2. Plot MFCC heatmap
def plot_mfcc(label):
    path = os.path.join(DATA_DIR, label)
    file = os.listdir(path)[1]  # pick second file
    file_path = os.path.join(path, file)
    
    signal, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCC - {label}")
    plt.tight_layout()
    plt.show()

# 3. Show label distribution
def plot_label_distribution():
    counts = {label: len(os.listdir(os.path.join(DATA_DIR, label))) for label in LABELS}
    plt.bar(counts.keys(), counts.values(), color=['red', 'blue'])
    plt.title("Label Distribution")
    plt.ylabel("Count")
    plt.show()

# Run all plots
for label in LABELS:
    plot_waveform(label)
    plot_mfcc(label)

plot_label_distribution()

# Build the model
model = models.Sequential()

# Add a 1D Convolutional layer
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(130, 13)))
model.add(layers.MaxPooling1D(2))

# Add another Convolutional layer
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))

# Flatten the output
model.add(layers.Flatten())

# Fully connected (Dense) layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))  # 2 classes: "up" and "down"

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Save the trained model
model.save('speech_music_player_model.h5')

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Function to test model with an audio file
def test_model_with_audio(file_path):
    # Load the audio file
    signal, sr = librosa.load(file_path, sr=16000)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    
    # Pad or truncate the MFCC to ensure it has exactly 130 frames (time steps)
    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, 130 - mfcc.shape[1]))), mode='constant')
    mfcc = mfcc[:, :130]  # Ensure it's exactly 130 time steps
    
    # Transpose the MFCC to match the input shape expected by the model (130, 13)
    mfcc = np.transpose(mfcc)  # Now it becomes (13, 130)
    
    # Reshape the MFCC to match the model's expected input shape (batch_size, 130, 13)
    mfcc = np.expand_dims(mfcc, axis=0)  # Make it a batch of size 1
    
    # Make prediction using the trained model
    prediction = model.predict(mfcc)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    
    # Return the predicted class label
    return predicted_class

# Path to the test audio file
test_file = 'myaudi.wav'  # Replace with the actual path to 'myaudio.wav'

# Run the test function
predicted_class = test_model_with_audio(test_file)

# Print the result
LABELS = ['down', 'up']  # Adjust if you have different label mapping
print(f"Predicted class for {test_file}: {LABELS[predicted_class]}")
