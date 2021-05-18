import os
import sys
import tqdm
import logging
import requests
from transformers import set_seed
from transformers import HfArgumentParser
from pytorch_lightning import _logger as lightning_logger


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
    },
    "klue-nli": {
        "train": {
            "googledrive_file_id": "18LhrHaPEW0VITMPfnwKXJ6bNuklBdi4U",
            "fname": "klue_nli_train.json",
        },
        "val": {
            "googledrive_file_id": "1UKIDAFOFuDSah7A66FZXSA8XUWUHhBAd",
            "fname": "klue_nli_dev.json",
        }
    },
    "ner": {
        "train": {
            "googledrive_file_id": "1RP764owqs1kZeHcjFnCX7zXt2EcjGY1i",
            "fname": "train.txt",
        },
        "val": {
            "googledrive_file_id": "1bEPNWT5952rD3xjg0LfJBy3hLHry3yUL",
            "fname": "val.txt",
        },
    },
    "korquad-v1": {
        "train": {
            "web_url": "https://korquad.github.io/dataset/KorQuAD_v1.0_train.json",
            "fname": "KorQuAD_v1.0_train.json",
        },
        "val": {
            "web_url": "https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json",
            "fname": "KorQuAD_v1.0_dev.json",
        }
    }
}

REMOTE_MODEL_MAP = {
    "kogpt2": {
        "merges": {
            "googledrive_file_id": "19-vpk-RAPhmIM1pPJ66F2Kbj4dW5V5sV",
            "fname": "merges.txt",
        },
        "vocab": {
            "googledrive_file_id": "19vjuxYOmlNTfg8kYKOPOUlZERm-QoTnj",
            "fname": "vocab.json",
        },
        "model": {
            "googledrive_file_id": "1dDGtsMy1NsfpuvgX8XobBsCYyctn5Xex",
            "fname": "pytorch_model.bin",
        },
        "config": {
            "googledrive_file_id": "1z6obNRWPHoVrMzT9THElblebdovuDLUZ",
            "fname": "config.json",
        },
    },
}
GOOGLE_DRIVE_URL = "https://docs.google.com/uc?export=download"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def save_response_content(response, save_path):
    with open(save_path, "wb") as f:
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
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


def download_downstream_dataset(args):
    data_name = args.downstream_corpus_name.lower()
    if data_name in REMOTE_DATA_MAP.keys():
        cache_dir = os.path.join(args.downstream_corpus_root_dir, data_name)
        for value in REMOTE_DATA_MAP[data_name].values():
            if "web_url" in value.keys():
                web_download(
                    url=value["web_url"],
                    save_fname=value["fname"],
                    cache_dir=cache_dir,
                    force_download=args.force_download,
                )
            else:
                google_download(
                    file_id=value["googledrive_file_id"],
                    save_fname=value["fname"],
                    cache_dir=cache_dir,
                    force_download=args.force_download
                )
    else:
        raise ValueError(f"not valid data name({data_name}), cannot download resources")


def download_pretrained_model(args, config_only=False):
    pretrained_model_name = args.pretrained_model_name.lower()
    if pretrained_model_name in REMOTE_MODEL_MAP.keys():
        for key, value in REMOTE_MODEL_MAP[pretrained_model_name].items():
            if not config_only or (config_only and key == "config"):
                if "web_url" in value.keys():
                    web_download(
                        url=value["web_url"],
                        save_fname=value["fname"],
                        cache_dir=args.pretrained_model_cache_dir,
                        force_download=args.force_download,
                    )
                else:
                    google_download(
                        file_id=value["googledrive_file_id"],
                        save_fname=value["fname"],
                        cache_dir=args.pretrained_model_cache_dir,
                        force_download=args.force_download,
                    )
    else:
        raise ValueError(f"not valid model name({pretrained_model_name}), cannot download resources")


def set_logger(args):
    # 파이토치 라이트닝이 학습 초반부에 로그를 두 번씩 찍는 버그가 있어
    # 이를 해결하기 위해 handler를 빈 리스트로 초기화한다
    lightning_logger.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Training/evaluation parameters %s", args)


def seed_setting(args):
    set_seed(args.seed)
    logger.info(f"complete setting seed({args.seed})")


def load_arguments(argument_class, json_file_path=None):
    parser = HfArgumentParser(argument_class)
    if json_file_path is not None:
        args, = parser.parse_json_file(json_file=json_file_path)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()
    return args