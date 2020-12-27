import torch
from tqdm import tqdm
from flask import Flask, request, jsonify, render_template


def encoding_passage(inference_dataloader, model, tokenizer):
    all_passages = []
    all_passage_embeddings = []
    special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    with torch.no_grad():
        for batch_inputs in tqdm(inference_dataloader, desc="passage encoding"):
            batch_input_ids = batch_inputs["input_ids"]
            for input_ids in batch_input_ids:
                input_ids = [el for el in input_ids if el not in special_tokens]
                all_passages.append(tokenizer.decode(input_ids))
            # passage_embeddings : args.batch_size x hidden_dimension
            passage_embeddings = model(**{**{f"passage_{k}": v for k, v in batch_inputs.items()}, "mode": "encoding"})
            all_passage_embeddings.append(passage_embeddings)
        # all passage_embeddings : total_num_of_passage x hidden_dimension
        all_passage_embeddings = torch.cat(all_passage_embeddings)
    return all_passages, all_passage_embeddings


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
        output_data = inference_fn(query["question"], query["context"])
        response = jsonify(output_data)
        return response

    return app
