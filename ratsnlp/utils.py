import os
import tqdm
import logging
import requests

REMOTE_DATA_MAP = {
    "kobert" : {
        "tokenizer" : {
            'url': 'https://kobert.blob.core.windows.net/models/kobert/tokenizer/kobert_news_wiki_ko_cased-ae5711deb3.spiece',
            'fname': 'vocab.txt',
        },
        "model" : {
            'url': 'https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params',
            'fname': 'pytorch_model.bin',
        },
        "config" : {
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
GOOGLE_URL = "https://docs.google.com/uc?export=download"
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


def google_download(file_id,
                    save_fname,
                    cache_dir="~/cache",
                    force_download=False):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    # 캐시 디렉토리 절대 주소 확인
    if cache_dir.startswith("~"):
        cache_dir = os.path.expanduser(cache_dir)
    else:
        cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    # 캐시 파일이 있으면 캐시 사용
    save_path = os.path.join(cache_dir, save_fname)
    if os.path.exists(save_path) and not force_download:
        logger.info(f"cache file({save_path}) exists, using cache!")
        return save_path
    # init a HTTP session
    session = requests.Session()
    # make a request
    response = session.get(GOOGLE_URL, params={'id': file_id}, stream=True)
    # get confirmation token
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(GOOGLE_URL, params=params, stream=True)
    # download to disk
    save_response_content(response, save_path)
    return save_path


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
    # 캐시 디렉토리 절대 주소 확인
    if cache_dir.startswith("~"):
        cache_dir = os.path.expanduser(cache_dir)
    else:
        cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    # 캐시 파일이 있으면 캐시 사용
    save_path = os.path.join(cache_dir, save_fname)
    if os.path.exists(save_path) and not force_download:
        logger.info(f"cache file({save_path}) exists, using cache!")
        return save_path
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
    save_response_content(response, save_path)
    return save_path
