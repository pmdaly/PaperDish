import os
import torch
from models import AlexNet, topk_to_rank_string
from helpers import allowed_file, valid_request
from flask import Flask, render_template, redirect, request, url_for, Markup
from PIL import Image
from werkzeug.utils import secure_filename


app = Flask(__name__)
UPLOAD_FOLDER = './static/images/'
'''
create folder if it doesn't exist
'''


@app.route('/index')
@app.route('/')
def index():
    return render_template('/pages/index.html')

@app.route('/alexnet', methods=['POST'])
def alexnet():
    if not valid_request(request, UPLOAD_FOLDER): return redirect(request.url)
    # hard coded image as orig.png
    model = AlexNet(pretrained=True)
    model.eval()

    image = Image.open(os.path.join(UPLOAD_FOLDER, 'orig.png'))
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
