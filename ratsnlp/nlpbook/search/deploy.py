import os
import time
import torch
import logging
from tqdm import tqdm
from flask import Flask, request, jsonify, render_template

logger = logging.getLogger(__name__)


def encoding_passage(inference_dataloader, model, tokenizer, args):
    passage_embedding_fpath = os.path.join(
            args.passage_embedding_dir,
            "passage-embeddings_{}_{}".format(
                tokenizer.__class__.__name__,
                str(args.passage_max_seq_length),
            ),
        )
    if os.path.exists(passage_embedding_fpath) and not args.overwrite_cache:
        start = time.time()
        all_passages, all_passage_embeddings = torch.load(passage_embedding_fpath, map_location=torch.device("cpu"))
        logger.info(
            f"Loading passage embeddings from cached file {passage_embedding_fpath} [took %.3f s]", time.time() - start
        )
    else:
        logger.info(f"Creating passage embeddings ...")
        all_passages = []
        all_passage_embeddings = []
        special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
        with torch.no_grad():
            for batch_inputs in tqdm(inference_dataloader, desc="passage encoding"):
                batch_input_ids = batch_inputs["input_ids"]
                for input_ids in batch_input_ids:
                    input_ids = [el for el in input_ids if el not in special_tokens]
                    all_passages.append(tokenizer.decode(input_ids))
                if torch.cuda.is_available():
                    batch_inputs = {k: v.cuda() for k, v in batch_inputs.items()}
                # passage_embeddings : args.batch_size x hidden_dimension
                passage_embeddings = model(**{**{f"passage_{k}": v for k, v in batch_inputs.items()}, "mode": "encoding"})
                all_passage_embeddings.append(passage_embeddings)
            # all passage_embeddings : total_num_of_passage x hidden_dimension
            all_passage_embeddings = torch.cat(all_passage_embeddings)
        start = time.time()
        logging.info(
            "Saving passage embeddings into cached file, it could take a lot of time..."
        )
        torch.save((all_passages, all_passage_embeddings), passage_embedding_fpath)
        logger.info(
            "Saving passage embeddings into cached file %s [took %.3f s]", passage_embedding_fpath, time.time() - start
        )
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
        output_data = inference_fn(query["question"])
        response = jsonify(output_data)
        return response

    return app
