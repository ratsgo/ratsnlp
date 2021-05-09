from flask import Flask, request, jsonify, render_template


def get_web_service_app(inference_fn, is_colab=True):

    app = Flask(__name__, template_folder='')
    if is_colab:
        from flask_ngrok import run_with_ngrok
        run_with_ngrok(app)
    else:
        from flask_cors import CORS
        CORS(app)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api', methods=['POST'])
    def api():
        query = request.json
        output_data = inference_fn(
            query["prompt"],
            query["min_length"],
            query["max_length"],
            query["top_p"],
            query["top_k"],
            query["repetition_penalty"],
            query["no_repeat_ngram_size"],
            query["temperature"],
        )
        response = jsonify(output_data)
        return response

    return app
