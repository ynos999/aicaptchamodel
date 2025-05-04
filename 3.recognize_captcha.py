import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import string
import sys
import os

# === Parametri ===
IMG_HEIGHT, IMG_WIDTH = 60, 250
CAPTCHA_LENGTH = 7
# CHARACTERS = string.ascii_lowercase + string.digits + string.punctuation
CHARACTERS = string.ascii_letters  # lielie + mazie burti
INV_CHAR_DICT = {i: c for i, c in enumerate(CHARACTERS)}

# === Ielādē modeli ===
MODEL_PATH = "captcha_model_final.keras"  # vai .h5 ja izmanto vecāku formātu
model = load_model(MODEL_PATH)

# === Attēla apstrāde ===
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
    except Exception as e:
        print(f"Kļūda atverot attēlu: {e}")
        return None
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
    return img_array

# === Prognozē CAPTCHA ===
def predict_captcha(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return None
    predictions = model.predict(img)
    predicted_indices = [np.argmax(p[0]) for p in predictions]
    predicted_text = ''.join(INV_CHAR_DICT[i] for i in predicted_indices)
    return predicted_text

# === Galvenais izpildījums ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Lietošana: python recognize_captcha.py <attēla_ceļš>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Fails nav atrasts: {image_path}")
        sys.exit(1)

    rezultats = predict_captcha(image_path)
    if rezultats:
        print("Atpazītais CAPTCHA teksts:", rezultats)
