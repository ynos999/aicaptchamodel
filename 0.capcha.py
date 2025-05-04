from captcha.image import ImageCaptcha
import random
import string
import os
import csv
import uuid

# Rakstzīmes: mazie un lielie burti
CHARACTERS = string.ascii_letters  # ietver gan lowercase, gan uppercase
# CHARACTERS = string.ascii_letters + string.digits + string.punctuation
CAPTCHA_LENGTH = 7
NUM_IMAGES = 2000
OUTPUT_DIR = "captcha_images/"
CSV_FILE = "labels.csv"
# FONT_PATH = "ComicNeue-Bold.ttf"  # <- pielāgotais fonts

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_gen = ImageCaptcha(width=250, height=60)
# Izveido ģeneratoru ar pielāgotu fontu
# image_gen = ImageCaptcha(width=250, height=60, fonts=[FONT_PATH])

with open(CSV_FILE, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label'])  # CSV galvene

    for _ in range(NUM_IMAGES):
        label = ''.join(random.choices(CHARACTERS, k=CAPTCHA_LENGTH))
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image_gen.write(label, filepath)
        writer.writerow([filename, label])

print(f"Ģenerēti {NUM_IMAGES} attēli un saglabāti mapē '{OUTPUT_DIR}' un failā '{CSV_FILE}'")
