from tokenizers import SentencePieceBPETokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


MASK_TOKEN = "<unused0>"


class KoGPT2Tokenizer(PreTrainedTokenizerFast):
    """
    reference:
        taeminlee/KoGPT-Transformers
        https://github.com/taeminlee/KoGPT2-Transformers/blob/master/kogpt2_transformers/kogpt2_transformers/tokenization_kogpt2.py
    """

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "special_tokens_map_file": "special_tokens_map.json",
        "added_tokens_file": "added_tokens.json"
    }

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_prefix_space=False,
        **kwargs
    ):
        super().__init__(
            SentencePieceBPETokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                add_prefix_space=add_prefix_space,
            ),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return cls._from_pretrained(*inputs, **kwargs)
