from typing import List

import torch
from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer
from torch import Tensor


class BeamSearchCERMetric(BaseMetric):
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
        cers = []
        predictions = log_probs.cpu()
        lengths = log_probs_length.detach().numpy()
        if self.lm:
            pred_texts = self.text_encoder.ctc_decode_beamsearch(
                predictions, self.beam_size, self.lm, lengths
            )
            for pred_text, target_text in zip(pred_texts, text):
                cers.append(calc_cer(target_text, pred_text[0]))
        else:
            predictions = predictions.numpy()
            for log_prob_vec, length, target_text in zip(predictions, lengths, text):
                target_text = self.text_encoder.normalize_text(target_text)
                pred_text = self.text_encoder.ctc_decode_beamsearch(
                    log_prob_vec[:length], self.beam_size, self.lm
                )
                cers.append(calc_cer(target_text, pred_text[0]))
        return sum(cers) / len(cers)


class ArgmaxCERMetric(BaseMetric):
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
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
