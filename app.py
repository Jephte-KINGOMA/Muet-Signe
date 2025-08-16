# app.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64
import uuid
import time

# Try tflite_runtime first (lightweight). If absent, fallback to tensorflow.lite.Interpreter
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_BACKEND = "tflite_runtime"
except Exception:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        TFLITE_BACKEND = "tensorflow"
    except Exception:
        Interpreter = None
        TFLITE_BACKEND = None

if Interpreter is None:
    raise RuntimeError("Aucun interpréteur TFLite trouvé. Installe 'tflite-runtime' ou 'tensorflow'.")

app = Flask(__name__)

# stockage temporaire en mémoire des images uplodées (image_id -> dict{bytes,ts})
UPLOADED_IMAGES = {}
MAX_IN_MEMORY = 200
IMAGE_EXPIRATION_SECONDS = 60 * 60  # 1 heure

# Chemin vers ton modèle TFLite (place signes_model.tflite à la racine du projet)
TFLITE_MODEL_PATH = "signes_model.tflite"

# Paramètres détection d'inconnues
CONFIDENCE_THRESHOLD = 0.80
MARGIN_THRESHOLD = 0.05

# Liste des classes (même ordre que la conversion)
CLASSES = [
    "1","2","3","4","5","6","7","8","9","10","A","Absent",
    "Activite","Aider","Aimer","Ami","Apre-Midi","Aujourd'hui","Avertir",
    "B","Bonjour","C","ca va ?","ca va mal","c'est clair","Collation",
    "Comment ca va?","Conger","Curiel","D","Demain","Desole","E","Enfant",
    "Et vous?","Excelent","F","Fete","Fin de la Semaine","Habiller","Heure",
    "Jamais","Manger","Matin","Merci","Message","Papier","Payer","Retard",
    "Signe","S'il vous plait","Sister","Special","Toilette","Toujours",
    "Travail","Urgent","Vacance"
]

# Chargement et préparation de l'interpréteur TFLite (global, une seule fois)
def load_tflite_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

if not os.path.exists(TFLITE_MODEL_PATH):
    raise FileNotFoundError(f"Fichier TFLite introuvable: {TFLITE_MODEL_PATH}. Place le fichier et redémarre.")

interpreter, input_details, output_details = load_tflite_interpreter(TFLITE_MODEL_PATH)

# Helpers
def cleanup_uploaded_images():
    now = time.time()
    to_delete = [k for k, v in UPLOADED_IMAGES.items() if now - v['ts'] > IMAGE_EXPIRATION_SECONDS]
    for k in to_delete:
        del UPLOADED_IMAGES[k]
    if len(UPLOADED_IMAGES) > MAX_IN_MEMORY:
        items = sorted(UPLOADED_IMAGES.items(), key=lambda kv: kv[1]['ts'])
        for k, _ in items[: len(UPLOADED_IMAGES) - MAX_IN_MEMORY]:
            del UPLOADED_IMAGES[k]

def read_image_from_bytes(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def tflite_predict(interpreter, input_details, output_details, img):
    """
    img : image BGR (cv2) already resized to expected size (H,W,3)
    returns: np.array of probabilities (1D)
    """
    # Determine expected input dtype/shape
    input_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape']  # e.g. [1,150,150,3]
    # prepare input array
    arr = np.expand_dims(img, axis=0)

    # If interpreter expects uint8, we pass uint8 values (0-255); else float32 normalized to [0,1]
    if input_dtype == np.uint8:
        arr_input = arr.astype(np.uint8)
    else:
        arr_input = arr.astype(np.float32) / 255.0

    # If model expects different input size, attempt to resize tensor (rare if fixed)
    try:
        # ensure correct shape: if first dimension is dynamic (0), resize
        if list(input_details[0]['shape']) != list(arr_input.shape):
            interpreter.resize_tensor_input(input_details[0]['index'], arr_input.shape)
            interpreter.allocate_tensors()
            # reload details
            # input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    except Exception:
        pass

    interpreter.set_tensor(input_details[0]['index'], arr_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # sometimes output is shape (1,n) -> return flatten
    return output_data.flatten()

@app.route('/', methods=['GET', 'POST'])
def index():
    image_data_url = None
    image_id = None
    result_text = ''

    cleanup_uploaded_images()

    if request.method == 'POST':
        # upload file (in-memory)
        file = request.files.get('user_image')
        if file and file.filename:
            img_bytes = file.read()
            if not img_bytes:
                result_text = "Fichier vide ou illisible."
            else:
                img = read_image_from_bytes(img_bytes)
                if img is None:
                    result_text = "Impossible de lire l'image téléchargée."
                else:
                    ext = os.path.splitext(secure_filename(file.filename))[1].lower()
                    if ext in ['.jpg', '.jpeg']:
                        mime = 'image/jpeg'
                    elif ext == '.png':
                        mime = 'image/png'
                    else:
                        mime = 'image/png'
                    b64 = base64.b64encode(img_bytes).decode('utf-8')
                    image_data_url = f"data:{mime};base64,{b64}"
                    image_id = str(uuid.uuid4())
                    UPLOADED_IMAGES[image_id] = {'bytes': img_bytes, 'ts': time.time()}

        # prediction
        if 'predict' in request.form:
            form_image_id = request.form.get('image_id')
            if not form_image_id:
                result_text = "Aucune image en mémoire. Veuillez télécharger une image."
            else:
                data = UPLOADED_IMAGES.get(form_image_id)
                if not data:
                    result_text = "Image expirée ou introuvable. Téléversez à nouveau."
                else:
                    img_bytes = data['bytes']
                    img = read_image_from_bytes(img_bytes)
                    if img is None:
                        result_text = "Impossible de lire l'image pour la prédiction."
                    else:
                        # prétraitement identique (resize + normalization inside tflite_predict)
                        img_resized = cv2.resize(img, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
                        preds = tflite_predict(interpreter, input_details, output_details, img_resized)
                        # assure probas (si modèle produit logits, user should convert; assume softmax output)
                        # sécurité: si somme différentes de 1, on fait softmax
                        s = np.sum(preds)
                        if not np.isclose(s, 1.0, atol=1e-3):
                            # applique softmax si pas normalisé
                            ex = np.exp(preds - np.max(preds))
                            preds = ex / np.sum(ex)

                        max_prob = float(np.max(preds))
                        idx = int(np.argmax(preds))
                        sorted_probs = np.sort(preds)[::-1]
                        diff_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])

                        if (max_prob < CONFIDENCE_THRESHOLD) or (diff_margin < MARGIN_THRESHOLD):
                            result_text = "Je ne connais pas cette image. Veuillez choisir une image de signes muets."
                        else:
                            result_text = CLASSES[idx]

                    b64 = base64.b64encode(img_bytes).decode('utf-8')
                    image_data_url = f"data:image/png;base64,{b64}"
                    image_id = form_image_id

    return render_template(
        'index.html',
        image_url = image_data_url,
        image_id = image_id,
        result_text = result_text
    )

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
