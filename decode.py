#!/usr/bin/env python3

from pathlib import Path

import torch
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from fireredasr.data.asr_feat import CMVN, ASRFeatExtractor
from fireredasr.models.fireredasr import FireRedAsr

"""
model args:
Namespace(blank='<blank>', blank_id=0, d_inner=5120, d_model=1280,
dropout_rate=0.1, eos='<eos>', eos_id=4, idim=80, input_length_max=60.0,
input_length_min=0.1, kernel_size=33, n_head=20, n_layers_dec=16,
n_layers_enc=16, odim=7832, output_length_max=250, output_length_min=1,
pad='<pad>', pad_id=2, pe_maxlen=5000, residual_dropout=0.1, sos='<sos>',
sos_id=3, unk='<unk>')

encoder parameters: 710543968, or 710.543968 million
decoder parameters: 429806080, or 429.80608 million
"""

from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf

import onnxruntime


class OnnxModel:
    def __init__(
        self,
        encoder: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.init_encoder(encoder)

    def init_encoder(self, encoder: str):
        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        print("---encoder input----")
        for i in self.encoder.get_inputs():
            print(i)

        print("---encoder output----")

        for i in self.encoder.get_outputs():
            print(i)
        print("------------")

    def run_encoder(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (N, T, C)
        """
        x_len = torch.tensor([x.shape[1]], dtype=torch.int64)
        encoder_out, encoder_out_len, encoder_mask = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
                self.encoder.get_outputs()[2].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_len.numpy(),
            },
        )
        return (
            torch.from_numpy(encoder_out),
            torch.from_numpy(encoder_out_len),
            torch.from_numpy(encoder_mask),
        )


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel

    # FireRedAsr requires un-normalized samples, e.g., samples in the range
    # [-32768, 32767]
    data = data * 32767
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_features(filename: str, dim: int = 80) -> torch.Tensor:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    wave, sample_rate = load_audio(filename)
    if sample_rate != 16000:
        import librosa

        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    features = []
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = dim
    opts.frame_opts.snip_edges = True
    fbank = knf.OnlineFbank(opts)

    fbank.accept_waveform(16000, wave)
    fbank.input_finished()
    for i in range(fbank.num_frames_ready):
        f = fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    return features


@torch.inference_mode()
def main():
    wave_filename = "./0.wav"
    features = compute_features(wave_filename)
    print(features.shape)  # (661, 80)

    cmvn = CMVN("/star-fj/fangjun/open-source/icefall-models/FireRedASR-AED-L/cmvn.ark")
    x = cmvn(features).float()

    d = "/star-fj/fangjun/open-source/icefall-models/FireRedASR-AED-L"
    model = FireRedAsr.from_pretrained("aed", d)
    model.model.eval()
    model.model.cpu()

    encoder_num_param = sum([p.numel() for p in model.model.encoder.parameters()])
    decoder_num_param = sum([p.numel() for p in model.model.decoder.parameters()])
    total_num_param = encoder_num_param + decoder_num_param
    print(
        f"encoder parameters: {encoder_num_param}, or {encoder_num_param/1000/1000} million"
    )
    print(
        f"decoder parameters: {decoder_num_param}, or {decoder_num_param/1000/1000} million"
    )
    print(
        f"total parameters: {total_num_param}, or {total_num_param/1000/1000} million, or {total_num_param/1000/1000/1000} billion"
    )

    onnx = OnnxModel(encoder="./onnx/encoder.int8.onnx")

    x = x.unsqueeze(0)
    use_onnx = True

    if use_onnx:
        enc_outputs, _, enc_mask = onnx.run_encoder(x)
        hyp = model.model.decoder.batch_beam_search(
            enc_outputs, enc_mask, 1, 1, 0, 1.0, 0.0, 1.0
        )[0][0]
    elif True:
        enc_outputs, _, enc_mask = model.model.encoder(
            x, torch.tensor([x.shape[1]], dtype=torch.int64)
        )
        hyp = model.model.decoder.batch_beam_search(
            enc_outputs, enc_mask, 1, 1, 0, 1.0, 0.0, 1.0
        )[0][0]
    else:
        hyp = model.model.transcribe(x, torch.tensor([x.shape[1]], dtype=torch.int64),)[
            0
        ][0]
    hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
    text = model.tokenizer.detokenize(hyp_ids)
    print(text)


if __name__ == "__main__":
    main()
