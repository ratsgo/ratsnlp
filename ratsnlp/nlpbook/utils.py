import os
import json
import tqdm
import logging
import requests
from transformers import set_seed


REMOTE_DATA_MAP = {
    "nsmc": {
        "train": {
            "web_url": "https://github.com/e9t/nsmc/raw/master/ratings_train.txt",
            "fname": "train.txt",
        },
        "val": {
            "web_url": "https://github.com/e9t/nsmc/raw/master/ratings_test.txt",
            "fname": "val.txt",
        },
    }
}

REMOTE_MODEL_MAP = {
    "kobert" : {
        "tokenizer" : {
            "googledrive_file_id": "1hNWzHMZcCkK50d_IqIN_ost8TZNg0lzL",
            "fname": "vocab.txt",
        },
        "model" : {
            "web_url": "https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params",
            "fname": "pytorch_model.bin",
        },
        "config" : {
            'model_type': "bert",
            'attention_probs_dropout_prob': 0.1,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'hidden_size': 768,
            'initializer_range': 0.02,
            'intermediate_size': 3072,
            'max_position_embeddings': 512,
            'num_attention_heads': 12,
            'num_hidden_layers': 12,
            'type_vocab_size': 2,
            'vocab_size': 8002,
        },
    },
}
GOOGLE_DRIVE_URL = "https://docs.google.com/uc?export=download"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def save_response_content(response, save_path):
    with open(save_path, "wb") as f:
        content_length = response.headers.get("Content-Length")
        total = int(content_length)
        progress = tqdm.tqdm(
            unit="B",
            unit_scale=True,
            total=total,
            initial=0,
            desc="Downloading",
            disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
        )
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                f.write(chunk)
        progress.close()


def get_valid_path(cache_dir, save_fname, make_dir=True):
    # 캐시 디렉토리 절대 주소 확인
    if cache_dir.startswith("~"):
        cache_dir = os.path.expanduser(cache_dir)
    else:
        cache_dir = os.path.abspath(cache_dir)
    if make_dir:
        os.makedirs(cache_dir, exist_ok=True)
    valid_save_path = os.path.join(cache_dir, save_fname)
    return valid_save_path


def google_download(file_id,
                    save_fname,
                    cache_dir="~/cache",
                    force_download=False):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    valid_save_path = get_valid_path(cache_dir, save_fname)
    # 캐시 파일이 있으면 캐시 사용
    if os.path.exists(valid_save_path) and not force_download:
        logger.info(f"cache file({valid_save_path}) exists, using cache!")
        return valid_save_path
    # init a HTTP session
    session = requests.Session()
    # make a request
    response = session.get(GOOGLE_DRIVE_URL, params={'id': file_id}, stream=True)
    # get confirmation token
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)
    # download to disk
    save_response_content(response, valid_save_path)
    return valid_save_path


def web_download(url,
                 save_fname,
                 cache_dir="~/cache",
                 proxies=None,
                 etag_timeout=10,
                 force_download=False):
    """
    download function. 허깅페이스와 SK T-BRAIN 다운로드 함수 참고.
    https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    https://github.com/SKTBrain/KoBERT/blob/master/kobert/utils.py
    """
    valid_save_path = get_valid_path(cache_dir, save_fname)
    # 캐시 파일이 있으면 캐시 사용
    if os.path.exists(valid_save_path) and not force_download:
        logger.info(f"cache file({valid_save_path}) exists, using cache!")
        return valid_save_path
    # url 유효성 체크
    # etag is None = we don't have a connection, or url doesn't exist, or is otherwise inaccessible.
    etag = None
    try:
        response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
        if response.status_code == 200:
            etag = response.headers.get("ETag")
    except (EnvironmentError, requests.exceptions.Timeout):
        pass
    if etag is None:
        raise ValueError(f"not valid URL({url}), cannot download resources")
    response = requests.get(url, stream=True)
    save_response_content(response, valid_save_path)
    return valid_save_path


def download_downstream_dataset(data_name, cache_dir="~/cache", force_download=False):
    data_name = data_name.lower()
    if data_name in REMOTE_DATA_MAP.keys():
        for value in REMOTE_DATA_MAP[data_name].values():
            if "web_url" in value.keys():
                web_download(value["web_url"], value["fname"], cache_dir, force_download=force_download)
            else:
                google_download(value["googledrive_file_id"], value["fname"], cache_dir, force_download)
    else:
        raise ValueError(f"not valid data name({data_name}), cannot download resources")


def download_pretrained_model(model_name, cache_dir="~/cache", force_download=False):
    model_name = model_name.lower()
    if model_name in REMOTE_MODEL_MAP.keys():
        for key, value in REMOTE_MODEL_MAP[model_name].items():
            if key != "config":
                if "web_url" in value.keys():
                    web_download(value["web_url"], value["fname"], cache_dir, force_download=force_download)
                else:
                    google_download(value["googledrive_file_id"], value["fname"], cache_dir, force_download)
            else:
                valid_save_path = get_valid_path(cache_dir, "config.json")
                if os.path.exists(valid_save_path) and not force_download:
                    logger.info(f"cache file({valid_save_path}) exists, using cache!")
                else:
                    with open(valid_save_path, "w") as f:
                        json.dump(value, f, indent=4)
    else:
        raise ValueError(f"not valid model name({model_name}), cannot download resources")


def set_logger(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Training/evaluation parameters %s", args)


def check_exist_checkpoints(args):
    if (
        os.path.exists(args.downstream_model_dir)
        and os.listdir(args.downstream_model_dir)
        and args.do_train
        and not args.overwrite_model
    ):
        raise ValueError(
            f"Output directory ({args.downstream_model_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    else:
        logger.info(f"Output directory ({args.downstream_model_dir}) is empty. check OK!")


def seed_setting(args):
    set_seed(args.seed)
    logger.info(f"complete setting seed({args.seed})")