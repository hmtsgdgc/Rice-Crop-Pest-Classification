import os
import shutil
import uuid
import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, session

# TensorFlow warnings ko chupane ke liye
os.environ['TF_CP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input

# _____________________________________________________________________________________________________________

app = Flask(__name__)

app.secret_key = 'kuch_bhi_secret_code'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#___________________________________________________________________________________________________________________
# load trained models

feature_extractor = tf.keras.models.load_model('static/models/efficientnetb1_feature_extractor.keras', compile = False)
scaler = joblib.load('static/models/efficientnetb1_svm_scaler.pkl')
svm_model = joblib.load('static/models/efficientnetb1_rbf_svm.pkl')

# class mapping
CLASSES = [
    'brown plant hopper',
      'paddy stem maggot',
      'rice borer',
      'rice gall midge',
      'rice leaf caterpillar',
     'rice leaf hopper',
     'rice leaf roller',
     'rice skipper',
     'rice water weevil',
      'rice whorl maggot',
       'thrips'
]

# ___________________________________________________________________________________________________________________
#  Image preprocessing

def preprocess_image(image_path):

    img = Image.open(image_path).convert('RGB')

    img = img.resize((240, 240))

    img_array = np.array(img, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

#  User folder

def get_user_folder():

    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    user_folder = os.path.join(
        app.config['UPLOAD_FOLDER'],
        session['user_id']
    )

    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    return user_folder

# 5. Clear previous files

def clear_user_folder(user_folder):

    if os.path.exists(user_folder):

        for filename in os.listdir(user_folder):

            file_path = os.path.join(user_folder, filename)

            try:

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')



@app.route('/', methods=['GET', 'POST'])
def home():

    prediction_name = None
    uploaded_image_name = None

    if request.method == 'POST':

        if 'file' in request.files:

            file = request.files['file']

            if file.filename != '':
                user_dir = get_user_folder()
                clear_user_folder(user_dir)
                filepath = os.path.join(user_dir, file.filename)

                file.save(filepath)

                uploaded_image_name = (
                    f"uploads/{session['user_id']}/{file.filename}"
                )

                try:
                    # Image preprocessing
                    processed_img = preprocess_image(filepath)

                    # Feature extraction
                    features = feature_extractor.predict(
                        processed_img,
                        verbose=0
                    )

                    # Scaling
                    scaled_features = scaler.transform(features)

                    # SVM prediction
                    predicted_class_idx = svm_model.predict(scaled_features)[0]

                    # Class name
                    prediction_name = CLASSES[int(predicted_class_idx)]

                except Exception as e:
                    prediction_name = f"Error: {str(e)}"
        
        return render_template('index.html', prediction=prediction_name, image_url=uploaded_image_name)
    else:
        return render_template('index.html')

# 6. Prediction route

# @app.route('/detection', methods=['GET', 'POST'])
# def detection():

#     prediction_name = None
#     uploaded_image_name = None

#     if request.method == 'POST':

#         if 'file' in request.files:

#             file = request.files['file']

#             if file.filename != '':
#                 user_dir = get_user_folder()
#                 clear_user_folder(user_dir)
#                 filepath = os.path.join(user_dir, file.filename)

#                 file.save(filepath)

#                 uploaded_image_name = (
#                     f"uploads/{session['user_id']}/{file.filename}"
#                 )

#                 try:
#                     # Image preprocessing
#                     processed_img = preprocess_image(filepath)

#                     # Feature extraction
#                     features = feature_extractor.predict(
#                         processed_img,
#                         verbose=0
#                     )

#                     # Scaling
#                     scaled_features = scaler.transform(features)

#                     # SVM prediction
#                     predicted_class_idx = svm_model.predict(scaled_features)[0]

#                     # Class name
#                     prediction_name = CLASSES[int(predicted_class_idx)]

#                 except Exception as e:
#                     prediction_name = f"Error: {str(e)}"

#     return render_template('detection.html', prediction=prediction_name, image_url=uploaded_image_name)



if __name__ == '__main__':
    app.run(debug = True)
