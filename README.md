## AI Captcha Model
### For Windows:
### python -m venv venv
### .\venv\Scripts\activate
### python.exe -m pip install --upgrade pip
### pip install -r requirements.txt

### python 0.capcha.py
### python 1.model.py
### python 2.precision.py
### python 3.recognize_captcha.py .\captcha_images\0c137a11d25b4f2db605ab2bf7ee37e4.png

#### Data loading:
#### Data is loaded from labels.csv, and is converted into an array of images (X) and the corresponding categorical formats (Y).

#### CaptchaDataGenerator:
#### The generator ensures that data processing is done in batches. Each batch contains images and the corresponding categorical labels for each character in the CAPTCHA text.

#### Model:
#### A CNN model is created with 2 convolutional and max pooling layers, followed by fully connected layers with dropout to prevent over-fitting.

#### Training and model:
#### Training is done with ModelCheckpoint to save the model with the best validation loss. After training, the model is saved as captcha_model_final.keras.

#### What needs to be done to make the script work:
#### Image and CSV files:
#### Make sure that the captcha_images/ folder contains all the images that match the entries in labels.csv.

#### Model training:
#### When you run this script, it will train the model, and save the best result (captcha_model_final.keras).

#### Testing:
#### Once the model is trained, you can use the model.predict() method to test how it performs with new CAPTCHA images.

----------------------------------------------------------
#### Datu ielāde:
#### Dati tiek ielādēti no labels.csv, un tiek pārveidoti attēlu masīvā (X) un atbilstošajos kategoriskajos formātos (Y).

#### CaptchaDataGenerator:
#### Ģenerators nodrošina, ka datu apstrāde tiek veikta partiju veidā. Katrs batch (partija) satur attēlus un atbilstošās kategoriskās etiķetes katram raksturam CAPTCHA tekstā.

#### Modelis:
#### Izveidots CNN modelis ar 2 konvolūcijas un maksimālās pooling slāņiem, seko pilnīgi savienoti slāņi (fully connected layers) ar dropout, lai novērstu pārmācīšanos.

#### Apmācība un modelis:
#### Apmācība tiek veikta ar ModelCheckpoint, lai saglabātu modeli ar labāko validācijas zaudējumu (loss). Pēc apmācības modelis tiek saglabāts kā captcha_model_final.keras.

#### Kas ir jādara, lai skripts darbotos:
#### Attēlu un CSV faili:
#### Pārliecinies, ka captcha_images/ mape satur visus attēlus, kas atbilst ierakstiem labels.csv.

#### Modeļa apmācība:
#### Kad tu palaiž šo skriptu, tas apmācīs modeli, un saglabās labāko rezultātu (captcha_model_final.keras).

#### Testēšana:
#### Kad modelis ir apmācīts, vari izmantot metodi model.predict(), lai pārbaudītu, kā tas darbojas ar jauniem CAPTCHA attēliem.