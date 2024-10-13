import re
from collections import defaultdict
from string import ascii_lowercase
from torchaudio.models.decoder import ctc_decoder

import torch

import requests
import gzip
import shutil
import os


# TODO add BPE

def download_model():
    url = 'http://www.openslr.org/resources/11/4-gram.arpa.gz'

    download_path = '4-gram.arpa.gz'
    extracted_file_path = 'lowercase_4-gram.arpa'

    print("Начинается скачивание файла с моделью")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, 'wb') as f:
            f.write(response.raw.read())
    else:
        print("Ошибка загрузки файла:", response.status_code)

    print("Начинаеncz распаковка файла")
    if os.path.exists(download_path):
        with gzip.open(download_path, 'rb') as f_in, open(extracted_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove("4-gram.arpa.gz")
        os.makedirs("data/sub_models/", exist_ok=True)
        os.rename(
            "lowercase_4-gram.arpa",
            "data/sub_models/lowercase_4-gram.arpa",
        )
        print(f"Файл {extracted_file_path} успешно создан.")
    else:
        print("Скачанный файл не найден.")


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        download_model()
        self.lm = "data/sub_models/lowercase_4-gram.arpa"
        self.bs_decoder = None

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            if last_char == self.ind2char[ind]:
                continue
            if self.ind2char[ind] != self.EMPTY_TOK:
                decoded.append(self.ind2char[ind])
            last_char = self.ind2char[ind]
        return "".join(decoded)

    def ctc_decode_beamsearch(self, probs, beam_size=100, with_lm=True, length=None) -> list[str]:
        if with_lm:
            if self.bs_decoder is None:
                self.bs_decoder = ctc_decoder(lexicon=None, tokens=self.vocab, lm=self.lm, nbest=3, beam_size=50, blank_token=self.EMPTY_TOK, sil_token=" ", lm_weight=0.5)
            bs_result = self.bs_decoder(torch.exp(probs).float(), length)
            result = []
            for pred in bs_result:
                res_str = self.decode(pred[0].tokens)
                result.append(res_str)
            return result
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for prob in probs:
            dp = self._expand_and_merge_path(dp, prob)
            dp = self._truncate_paths(dp, beam_size)

        dp = [prefix for (prefix, _), _ in sorted(dp.items(), key=lambda x: -x[1])]
        return dp

    def _expand_and_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                elif cur_char != self.EMPTY_TOK:
                    new_prefix = prefix + cur_char
                else:
                    new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp

    def _truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
