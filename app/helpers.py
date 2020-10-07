import os
from flask import flash

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def valid_request(request, upload_folder):
    if 'file' not in request.files:
        flash('No file part')
        return False
    image = request.files['file']
    if image.filename == '':
        flash('No selected file')
        return False
    if image and allowed_file(image.filename):
        # do i need this?
        # image_fn = secure_filename(image.filename)
        image.save(os.path.join(upload_folder, 'orig.png'))
        return True
