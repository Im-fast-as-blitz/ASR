from typing import List

import torch
from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer
from torch import Tensor


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=10, lm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.lm = lm

    def __call__(
        self,
        log_probs: Tensor,
        log_probs_length: Tensor,
        text: List[str],
        **kwargs,
    ):
        wers = []
        predictions = log_probs.cpu().numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode_beamsearch(
                log_prob_vec[:length], self.beam_size, self.lm
            )

            wers.append(calc_wer(target_text, pred_text[0]))
        return sum(wers) / len(wers)


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self,
        log_probs: Tensor,
        log_probs_length: Tensor,
        text: List[str],
        **kwargs,
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
