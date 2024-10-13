import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.utils.io_utils import read_json


def main(dir_path):
    wer_sum = 0
    cer_sum = 0
    count = 0
    for path in Path(dir_path).iterdir():
        data = read_json(path)
        target_text = CTCTextEncoder.normalize_text(data["text"])
        predicted_text = data["pred_text"]
        wer_sum += calc_wer(target_text, predicted_text)
        cer_sum += calc_cer(target_text, predicted_text)
        count += 1
    print("WER:", wer_sum / count)
    print("CER:", cer_sum / count)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dir_path", default="data/saved/predict", type=str)
    args = args.parse_args()
    main(args.dir_path)