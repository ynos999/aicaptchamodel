import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import string

# Parametri
IMG_HEIGHT, IMG_WIDTH = 60, 250
CAPTCHA_LENGTH = 7
# CHARACTERS = string.ascii_lowercase + string.digits + string.punctuation
CHARACTERS = string.ascii_letters
NUM_CLASSES = len(CHARACTERS)
CHAR_DICT = {c: i for i, c in enumerate(CHARACTERS)}
INV_CHAR_DICT = {i: c for c, i in CHAR_DICT.items()}

IMAGE_DIR = "captcha_images/"
LABEL_FILE = "labels.csv"
MODEL_PATH = "captcha_model_final.keras"  # vai 'captcha_model_final.h5'

# Funkcija: tekstu pārvērš par skaitļiem
def text_to_labels(text):
    return [CHAR_DICT[c] for c in text]

# Funkcija: skaitļus atpakaļ uz tekstu
def labels_to_text(label_indices):
    return ''.join(INV_CHAR_DICT[i] for i in label_indices)

# Datu ielāde
def load_data(image_dir, label_file):
    df = pd.read_csv(label_file)
    X, y = [], []

    for _, row in df.iterrows():
        label = row['label']
        if len(label) != CAPTCHA_LENGTH:
            continue
        filepath = os.path.join(image_dir, row['filename'])
        try:
            img = Image.open(filepath).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img) / 255.0
            X.append(img_array)
            y.append(text_to_labels(label))
        except Exception as e:
            print(f"Kļūda ar {filepath}: {e}")
            continue

    X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    y = np.array(y)
    y_split = [to_categorical(y[:, i], num_classes=NUM_CLASSES) for i in range(CAPTCHA_LENGTH)]
    return X, y_split, y

# === Ielādē datus un modeli ===
X, y_split, y_true = load_data(IMAGE_DIR, LABEL_FILE)
model = load_model(MODEL_PATH)

# === Novērtē ===
results = model.evaluate(X, y_split, verbose=1)
for i in range(CAPTCHA_LENGTH):
    print(f"Rakstzīmes {i+1} precizitāte: {results[i+1]:.4f}")

# === Precizitātes pārbaude pilnam vārdiņam (string) ===
y_pred = model.predict(X)
y_pred_labels = np.stack([np.argmax(p, axis=1) for p in y_pred], axis=1)
y_true_labels = y_true

correct = 0
for i in range(len(y_true_labels)):
    if np.array_equal(y_pred_labels[i], y_true_labels[i]):
        correct += 1

full_string_accuracy = correct / len(y_true_labels)
print(f"\nPilna CAPTCHA teksta precizitāte: {full_string_accuracy:.4f}")
