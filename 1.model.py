import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import string

# Parametri
# IMG_HEIGHT, IMG_WIDTH = 50, 200
IMG_HEIGHT, IMG_WIDTH = 60, 250
CAPTCHA_LENGTH = 7
# CHARACTERS = string.ascii_lowercase + string.digits + string.punctuation
CHARACTERS = string.ascii_letters  # lielie + mazie burti

NUM_CLASSES = len(CHARACTERS)
CHAR_DICT = {c: i for i, c in enumerate(CHARACTERS)}
IMAGE_DIR = "captcha_images/"
LABEL_FILE = "labels.csv"

# Funkcija, lai pārveidotu tekstu par skaitliskām etiķetēm
def text_to_labels(text):
    return [CHAR_DICT[c] for c in text]

# Datu ielāde no CSV faila un attēlu sagatavošana
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
    return X, y_split

# Datu ģenerators, lai veiktu apmācību partiju veidā
# Ja testē ar nelielu attēlu skaitu (mazāk par 100), iestati batch_size=8 vai batch_size=16:
class CaptchaDataGenerator(Sequence):
    def __init__(self, df, image_dir, char_dict, characters, captcha_length=7, 
                 batch_size=32, img_height=60, img_width=250, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.char_dict = char_dict
        self.characters = characters
        self.num_classes = len(characters)
        self.captcha_length = captcha_length
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[idxs]

        X = np.zeros((self.batch_size, self.img_height, self.img_width, 1), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.num_classes), dtype=np.float32) for _ in range(self.captcha_length)]

        for i, (_, row) in enumerate(batch_df.iterrows()):
            filepath = os.path.join(self.image_dir, row['filename'])
            label = row['label']
            try:
                img = Image.open(filepath).convert('L').resize((self.img_width, self.img_height))
                img_array = np.array(img) / 255.0
                X[i, :, :, 0] = img_array
                for j, char in enumerate(label):
                    y[j][i, self.char_dict[char]] = 1.0
            except Exception as e:
                print(f"Kļūda ar {filepath}: {e}")
                continue

        # Pārliecināmies, ka X ir numpy masīvs un y ir tuple
        return np.array(X), tuple(np.array(y_i) for y_i in y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Modeļa definēšana
def build_model():
    input_layer = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # 7 izvades slāņi, katrs ar savu 'accuracy' metriku
    outputs = [layers.Dense(NUM_CLASSES, activation='softmax', name=f'char_{i}')(x)
               for i in range(CAPTCHA_LENGTH)]

    model = models.Model(inputs=input_layer, outputs=outputs)
    
    # Norādīt metriku katram izvades slānim
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[['accuracy']]*CAPTCHA_LENGTH)  # Pielāgota metrika katram izvades slānim

    return model


# Galvenais apmācības skripts
if __name__ == "__main__":
    # Ielādē datus
    df = pd.read_csv(LABEL_FILE)
    train_df, val_df = train_test_split(df, test_size=0.1)

    # Ģeneratori
    train_gen = CaptchaDataGenerator(train_df, IMAGE_DIR, CHAR_DICT, CHARACTERS)
    val_gen = CaptchaDataGenerator(val_df, IMAGE_DIR, CHAR_DICT, CHARACTERS)

    # Modeļa izveidošana
    model = build_model()

    # Modela apmācība ar saglabāšanu, kad uzlabojas validācijas precizitāte
    
    # checkpoint = ModelCheckpoint("captcha_model.h5", save_best_only=True, monitor="val_loss", mode="min", verbose=1)
    checkpoint = ModelCheckpoint("captcha_model.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=1)
    model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[checkpoint])

    # Modela saglabāšana
    # model.save("captcha_model_final.h5")
    model.save("captcha_model_final.keras")

    # Vai:
    # from keras.saving import save_model
    # save_model(model, "captcha_model_final.keras")

    print("Modelis ir apmācīts un saglabāts.")
