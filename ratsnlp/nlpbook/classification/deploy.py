from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify, render_template


def get_web_service_app(inference_fn):

    app = Flask(__name__, template_folder='')
    run_with_ngrok(app)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api', methods=['POST'])
    def api():
        query_sentence = request.json
        output_data = inference_fn(query_sentence)
        response = jsonify(output_data)
        return response

    return app
