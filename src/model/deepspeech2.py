from torch import nn
from torch.nn import Sequential


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout_rate=0,
        bidirectional=True,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional,
        )
        self.batch_norm = nn.BatchNorm1d(input_size)

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

    def forward(self, x, lengths):
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2).contiguous()
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )  # need N x T x *
        x, _ = self.gru(x, None)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        if self.bidirectional:
            x = x[..., : self.hidden_size] + x[..., self.hidden_size :]
        return x


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, fc_hidden, gru_count=7):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.conv_lay = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                32,
                96,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(
                96,
                96,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        n_feats = self.transform_input_lengths(n_feats)
        self.gru = nn.Sequential(GRU(96 * n_feats, fc_hidden))
        for i in range(gru_count - 1):
            self.gru.append(GRU(fc_hidden, fc_hidden))

        self.batch_norm = nn.BatchNorm1d(fc_hidden)
        self.final = nn.Linear(fc_hidden, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram = spectrogram.unsqueeze(1)
        output = self.conv_lay(spectrogram)

        b, c, f, t = output.shape
        output = output.view(b, c * f, t).transpose(1, 2)

        log_probs_length = self.transform_input_lengths(spectrogram_length)
        for gru in self.gru:
            output = gru(output, log_probs_length)

        output = self.batch_norm(output.transpose(1, 2)).transpose(1, 2).contiguous()
        output = self.final(output)

        log_probs = nn.functional.log_softmax(output, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        output_lengths = (input_lengths + 20 * 2 - 41) // 2 + 1
        output_lengths = (output_lengths + 10 * 2 - 21) // 2 + 1
        output_lengths = (output_lengths + 10 * 2 - 21) // 2 + 1
        return output_lengths

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
