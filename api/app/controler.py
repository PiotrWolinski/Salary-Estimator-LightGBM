import os

import requests
from flask import request
from werkzeug.utils import secure_filename

from app import app


@app.route('/api/upload', methods=['GET'])
def upload_file():
    f = request.files['image']
    save_filename = secure_filename(f.filename)
    f.save(os.path.join(UPLOAD_PATH, save_filename))
    return 'file uploaded successfully'
