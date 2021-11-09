# -*- coding: utf-8 -*-

from flasgger import Swagger
from flask import Flask, request, jsonify

import predict

swagger_config = Swagger.DEFAULT_CONFIG
swagger_config['swagger_ui_bundle_js'] = '//unpkg.com/swagger-ui-dist@3.3.0/swagger-ui-bundle.js'
swagger_config['swagger_ui_standalone_preset_js'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui-standalone-preset.js'
swagger_config['jquery_js'] = '//unpkg.com/jquery@2.2.4/dist/jquery.min.js'
swagger_config['swagger_ui_css'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui.css'

app = Flask(__name__)


def create_swagger(app_):
    template = {
        "openapi": '3.0.0',
        "info": {
            "title": "Automatic Image Assessment API by NiNe Capture",
            "version": "3.0",
            "description": "Here at NiNe Capture, we are passionate about pioneering cutting-edge technologies to maximize both the creativity and productivity of our services. In this API, we leverage deep convolutional neural networks to automatically evaluate and prioritize client photos. Having been trained on millions of high-quality open-source photographs and fine-tuned with thousands our own best works, our models are capable of processing and evaluating both the technical and aesthetic merit of thousands of photographs in a matter of seconds. This ensures that our clients receive unparalleled, high-quality work in record time."
        },
        "basePath": "/",
        "schemes": [
            "http",
            "https"
        ]
    }
    return Swagger(app_, template=template)


swag = create_swagger(app)


@app.route('/')
def index():
    from flask import request
    return f"Please refer to the <a href=\"{request.url_root}apidocs\">API documentation</a> for more details."


@app.route('/api/rank/arw/combined', methods=["GET"])
def arw_combined():
    """Predict scores for arw files using a combination of both aesthetic and technical models.
    Press the `Try it out` button below to test out sample data against the prediction REST API.
    ---
    tags:
      - name: Image Assessment
    parameters:
      - name: base_model_name
        in: query
        type: string
        required: true
        description: Base model specification
        default: 'MobileNet'

      - name: weights_aesthetic
        in: query
        type: string
        required: true
        description: Weights file for the aesthetic model
        default: 'weights_aesthetic.hdf5'

      - name: weights_technical
        in: query
        type: string
        required: true
        description: Weights file for the technical model
        default: 'weights_technical.hdf5'

      - name: directory
        in: query
        type: string
        required: true
        description: Path to image directory
        default: 'C:\\Dropbox\\NineCapture\\<date>\\<folder>'

      - name: extension
        in: query
        type: string
        required: true
        description: Extension of the image files
        default: 'ARW'
    responses:
        200:
            description: The output values of the request
    """
    # Return GET output if called directly from the REST API.
    if request.method == 'GET':
        prediction_dict = predict.rank_img_combo(
            base_model_name=request.args.get("base_model_name"),
            weights_aesthetic=request.args.get("weights_aesthetic"),
            weights_technical=request.args.get("weights_technical"),
            img_dir=request.args.get("directory"),
            ext=request.args.get("extension")
        )
        return jsonify(prediction_dict)


@app.route('/api/rank/arw/aesthetic', methods=["GET"])
def arw_aesthetic():
    """Predict scores for arw files using an aesthetic model.
    Press the `Try it out` button below to test out sample data against the prediction REST API.
    ---
    tags:
      - name: Image Assessment
    parameters:
      - name: base_model_name
        in: query
        type: string
        required: true
        description: Base model specification
        default: 'MobileNet'

      - name: weights_aesthetic
        in: query
        type: string
        required: true
        description: Weights file for the aesthetic model
        default: 'weights_aesthetic.hdf5'

      - name: directory
        in: query
        type: string
        required: true
        description: Path to image directory
        default: 'C:\\Dropbox\\NineCapture\\<date>\\<folder>'

      - name: extension
        in: query
        type: string
        required: true
        description: Extension of the image files
        default: 'ARW'
    responses:
        200:
            description: The output values of the request
    """
    # Return GET output if called directly from the REST API.
    if request.method == 'GET':
        predict.undo_rename_img(
            img_dir=request.args.get("directory"),
            ext=request.args.get("extension"))
        prediction_dict = predict.rank_img_aesthetic(
            base_model_name=request.args.get("base_model_name"),
            weights_aesthetic=request.args.get("weights_aesthetic"),
            img_dir=request.args.get("directory"),
            ext=request.args.get("extension"))
        return jsonify(prediction_dict)


@app.route('/api/rank/jpg/combined', methods=["GET"])
def jpg_combined():
    """Predict scores for jpg files using a combination of both aesthetic and technical models.
    Press the `Try it out` button below to test out sample data against the prediction REST API.
    ---
    tags:
      - name: Image Assessment
    parameters:
      - name: base_model_name
        in: query
        type: string
        required: true
        description: Base model specification
        default: 'MobileNet'

      - name: weights_aesthetic
        in: query
        type: string
        required: true
        description: Weights file for the aesthetic model
        default: 'weights_aesthetic.hdf5'

      - name: weights_technical
        in: query
        type: string
        required: true
        description: Weights file for the technical model
        default: 'weights_technical.hdf5'

      - name: directory
        in: query
        type: string
        required: true
        description: Path to image directory
        default: 'C:\\Dropbox\\NineCapture\\<date>\\<folder>'

      - name: extension
        in: query
        type: string
        required: true
        description: Extension of the image files
        default: 'jpg'
    responses:
        200:
            description: The output values of the request
    """
    # Return GET output if called directly from the REST API.
    if request.method == 'GET':
        prediction_dict = predict.rank_img_combo(
            base_model_name=request.args.get("base_model_name"),
            weights_aesthetic=request.args.get("weights_aesthetic"),
            weights_technical=request.args.get("weights_technical"),
            img_dir=request.args.get("directory"),
            ext=request.args.get("extension")
        )
        return jsonify(prediction_dict)


@app.route('/api/rank/jpg/aesthetic', methods=["GET"])
def jpg_aesthetic():
    """Predict scores for jpg files using an aesthetic model.
    Press the `Try it out` button below to test out sample data against the prediction REST API.
    ---
    tags:
      - name: Image Assessment
    parameters:
      - name: base_model_name
        in: query
        type: string
        required: true
        description: Base model specification
        default: 'MobileNet'

      - name: weights_aesthetic
        in: query
        type: string
        required: true
        description: Weights file for the aesthetic model
        default: 'weights_aesthetic.hdf5'

      - name: directory
        in: query
        type: string
        required: true
        description: Path to image directory
        default: 'C:\\Dropbox\\NineCapture\\<date>\\<folder>'

      - name: extension
        in: query
        type: string
        required: true
        description: Extension of the image files
        default: 'jpg'

    responses:
        200:
            description: The output values of the request
    """
    predict.undo_rename_img(
        img_dir=request.args.get("directory"),
        ext='jpg')
    # Return GET output if called directly from the REST API.
    if request.method == 'GET':
        prediction_dict = predict.jpg_aesthetic(
            base_model_name=request.args.get("base_model_name"),
            weights_aesthetic=request.args.get("weights_aesthetic"),
            jpg_dir=request.args.get("directory")
        )
        return jsonify(prediction_dict)


@app.route('/api/utils/unrename', methods=["GET"])
def rename_undo():
    """Undo the renaming of files to revert to the original file names.
    Press the `Try it out` button below to test out sample data against the prediction REST API.
    ---
    tags:
      - name: Image Utilities
    parameters:
      - name: directory
        in: query
        type: string
        required: true
        description: Path to image directory
        default: 'C:\\Dropbox\\NineCapture\\<date>\\<folder>'
      - name: extension
        in: query
        type: string
        required: true
        description: File type of the images
        default: 'ARW'
    responses:
        200:
            description: The output values of the request
    """
    if request.method == 'GET':
        predict.undo_rename_img(
            img_dir=request.args.get("directory"),
            ext=request.args.get("extension"))
        return jsonify({"success": True})


@app.route('/api/utils/dedupe', methods=["GET"])
def remove_duplicates():
    """Removes images that are similar to one another while keeping the higher rank.
    Press the `Try it out` button below to test out sample data against the prediction REST API.
    ---
    tags:
      - name: Image Utilities
    parameters:
      - name: directory
        in: query
        type: string
        required: true
        description: Path to image directory
        default: 'C:\\Dropbox\\NineCapture\\<date>\\<folder>'
    responses:
        200:
            description: The output values of the request
    """
    if request.method == 'GET':
        duplicates = predict.remove_duplicates(
            img_dir=request.args.get("directory"),
        )
        return jsonify({"success": True})


@app.route('/api/utils/demangle', methods=["GET"])
def demangle():
    """Cleans filenames for images
    Press the `Try it out` button below to test out sample data against the prediction REST API.
    ---
    tags:
      - name: Image Utilities
    parameters:
      - name: directory
        in: query
        type: string
        required: true
        description: Path to image directory
        default: 'C:\\Dropbox\\NineCapture\\<date>\\<folder>'
      - name: extension
        in: query
        type: string
        required: true
        description: Path to image directory
        default: 'jpg'
    responses:
        200:
            description: The output values of the request
    """
    if request.method == 'GET':
        predict.demangle_filenames(
            img_dir=request.args.get("directory"),
            ext=request.args.get("extension")
        )
        return jsonify({"success": True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
