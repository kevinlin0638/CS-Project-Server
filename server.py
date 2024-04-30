import json
import os
from flask import Flask, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

from portrait_server import generate_image

UPLOAD_FOLDER = './uploaded'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_folder='public', static_url_path='/img')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return {'status': 'success', 'filename': filename}
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/generate', methods=['GET', 'POST'])
@cross_origin()
def generate():
    if request.method == 'POST':
        data = json.loads(request.data)
        print(data)
        generate_image(data['prompt'], data['shape'], data['fileList'], data['uid'])
        return {'status':'success'}

    return ''''''
