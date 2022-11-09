from datetime import datetime
import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import Classifier
import shutil

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
UPLOAD_FOLDER = './uploads'
TEMP_FOLDER = './TEMP'
    
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER

def is_file_extension_valid(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate(file):
    if not file:
        return False, jsonify({"response": "failed", "message": "File not found!"})
    if not file.filename and is_file_extension_valid(file.filename):
        return False, jsonify({"response": "failed", "message": "File format should be valid!"})
    return True, None


@cross_origin()
@app.route('/api/v1/upload', methods=['POST'])
def upload_file():
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

    file = request.files.get("image")
    valid, error = validate(file)
    if not valid:
        return error

    file_name = "{datetime}.{fileExt}".format(datetime=datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), fileExt="jpeg")
    file_path = os.path.join(app.config['TEMP_FOLDER'], file_name)
    file.save(file_path)

    category, result = Classifier.classify(file_path)
    print(category)
    print(result)

    category_folder = os.path.join(app.config['UPLOAD_FOLDER'], category)
    os.makedirs(category_folder, exist_ok=True)

    shutil.copy(file_path, f"{category_folder}/{file_name}")
    shutil.rmtree(app.config['TEMP_FOLDER'])
    
    return jsonify({"response": "success", "message": "File uploaded successfully"})

@cross_origin()
@app.route('/', methods=['GET'])
def index():   
    return jsonify({"response": "success", "message": "Server Started"})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=90, debug=True)

     
