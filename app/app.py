import os
import torch
from models import AlexNet, topk_to_rank_string
from flask import Flask, flash, render_template, redirect, request, url_for , Markup
from PIL import Image
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
'''
create folder if it doesn't exist
'''

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def valid_request(request):
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
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], 'orig.png'))
        return True

@app.route('/index')
@app.route('/')
def index():
    return render_template('/pages/index.html')

@app.route('/alexnet', methods=['POST'])
def alexnet():
    if not valid_request(request): return redirect(request.url)
    # hard coded image as orig.png
    model = AlexNet(pretrained=True)
    model.eval()

    image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], 'orig.png'))
    image = image.convert('RGB')

    out = topk_to_rank_string(*model(image))

    return render_template('/pages/alexnet.html', out=Markup(out[:-2]))

@app.route('/fcn', methods=['POST'])
def fcn():
    if not valid_request(request): return redirect(request.url)
    # run classification on orig image
    return render_template('/pages/alexnet.html')

@app.route('/vgg', methods=['POST'])
def vgg():
    if not valid_request(request): return redirect(request.url)
    # run classification on orig image
    return render_template('/pages/alexnet.html')

@app.route('/alexnet', methods=['GET'])
@app.route('/fcn', methods=['GET'])
@app.route('/vgg', methods=['GET'])
def image_get():
    return render_template('/pages/image_form.html')

if __name__ == '__main__':
    app.run(debug=True, port=2222)
