import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch_size = len(dataset_items)

    # chaneles= dataset_items[0]["spectrogram"].shape[0]
    # print(chaneles) ==1

    # audio
    audio_size = [item["audio"].shape[1] for item in dataset_items]
    max_audio_size = max(audio_size)

    # spec
    spec_freq = dataset_items[0]["spectrogram"].shape[1]
    spec_size = [item["spectrogram"].shape[2] for item in dataset_items]
    max_spec_size = max(spec_size)

    # text
    text = [item["text"] for item in dataset_items]

    # text enc
    text_enc_size = [item["text_encoded"].shape[1] for item in dataset_items]
    max_text_enc_size = max(text_enc_size)

    # audio path
    audios_path = [item["audio_path"] for item in dataset_items]

    spec_batch = torch.zeros(size=(batch_size, spec_freq, max_spec_size))
    text_batch = torch.zeros(size=(batch_size, max_text_enc_size))
    audio_batch = torch.zeros(size=(batch_size, max_audio_size))
    for i in range(batch_size):
        spec_batch[i, ..., : spec_size[i]] = dataset_items[i]["spectrogram"].squeeze(0)
        text_batch[i, ..., : text_enc_size[i]] = dataset_items[i][
            "text_encoded"
        ].squeeze(0)
        audio_batch[i, : audio_size[i]] = dataset_items[i]["audio"].squeeze(0)

    result = {
        "audio": audio_batch,
        "audio_path": audios_path,
        "spectrogram": spec_batch,
        "spectrogram_length": torch.tensor(spec_size),
        "text": text,
        "text_encoded": text_batch,
        "text_encoded_length": torch.tensor(text_enc_size),
    }
    return result
