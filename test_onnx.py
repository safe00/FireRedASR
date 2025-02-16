#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf

"""
{'comment': 'This is FireRedASR-AED-L', 'url-2': 'https://huggingface.co/FireRedTeam/FireRedASR-AED-L', 'sos': '3', 'eos': '4', 'head_dim': '64', 'num_head': '20', 'version': '1', 'maintainer': 'k2-fsa', 'model_type': 'fire-red-asr-aed', 'model_author': 'FireRedTeam', 'n_mels': '80', 'url': 'https://github.com/FireRedTeam/FireRedASR', 'num_decoder_layers': '16'}
---encoder input----
NodeArg(name='x', type='tensor(float)', shape=[1, 'T', 80])
NodeArg(name='x_len', type='tensor(int64)', shape=[1])
---encoder output----
NodeArg(name='n_layer_cross_k', type='tensor(float)', shape=[16, 'Concatn_layer_cross_k_dim_1', 'T', 1280])
NodeArg(name='n_layer_cross_v', type='tensor(float)', shape=[16, 'Concatn_layer_cross_v_dim_1', 'T', 1280])
------------
---decoder input----
NodeArg(name='tokens', type='tensor(int64)', shape=[1, 1])
NodeArg(name='in_n_layer_self_k_cache', type='tensor(float)', shape=[16, 1, 1024, 20, 64])
NodeArg(name='in_n_layer_self_v_cache', type='tensor(float)', shape=[16, 1, 1024, 20, 64])
NodeArg(name='n_layer_cross_k', type='tensor(float)', shape=[16, 1, 'T', 1280])
NodeArg(name='n_layer_cross_v', type='tensor(float)', shape=[16, 1, 'T', 1280])
NodeArg(name='offset', type='tensor(int64)', shape=[1])
---decoder output----
NodeArg(name='logits', type='tensor(float)', shape=['MatMullogits_dim_0', 'MatMullogits_dim_1', 7832])
NodeArg(name='out_n_layer_self_k_cache', type='tensor(float)', shape=[16, 1, 1024, 20, 64])
NodeArg(name='out_n_layer_self_v_cache', type='tensor(float)', shape=[16, 1, 1024, 20, 64])
------------
"""


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.init_encoder(encoder)
        self.init_decoder(decoder)

    def init_encoder(self, encoder: str):
        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map

        self.num_decoder_layers = int(meta["num_decoder_layers"])
        self.num_head = int(meta["num_head"])
        self.head_dim = int(meta["head_dim"])
        self.sos = int(meta["sos"])
        self.eos = int(meta["eos"])
        self.max_len = int(meta["max_len"])
        self.cmvn_mean = np.array(
            list(map(float, meta["cmvn_mean"].split(","))), dtype=np.float32
        )
        self.cmvn_inv_stddev = np.array(
            list(map(float, meta["cmvn_inv_stddev"].split(","))), dtype=np.float32
        )

        print("---encoder input----")
        for i in self.encoder.get_inputs():
            print(i)

        print("---encoder output----")

        for i in self.encoder.get_outputs():
            print(i)
        print("------------")

    def init_decoder(self, decoder: str):
        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        print("---decoder input----")
        for i in self.decoder.get_inputs():
            print(i)

        print("---decoder output----")

        for i in self.decoder.get_outputs():
            print(i)
        print("------------")

    def get_self_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = 1
        n_layer_self_k_cache = np.zeros(
            (
                self.num_decoder_layers,
                batch_size,
                self.max_len,
                self.num_head,
                self.head_dim,
            ),
            dtype=np.float32,
        )
        n_layer_self_v_cache = np.zeros(
            (
                self.num_decoder_layers,
                batch_size,
                self.max_len,
                self.num_head,
                self.head_dim,
            ),
            dtype=np.float32,
        )
        return n_layer_self_k_cache, n_layer_self_v_cache

    def run_encoder(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_len = np.array([x.shape[1]], dtype=np.int64)
        n_layer_cross_k, n_layer_cross_v = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x,
                self.encoder.get_inputs()[1].name: x_len,
            },
        )
        return n_layer_cross_k, n_layer_cross_v

    def run_decoder(
        self,
        tokens: np.ndarray,
        n_layer_self_k_cache: np.ndarray,
        n_layer_self_v_cache: np.ndarray,
        n_layer_cross_k: np.ndarray,
        n_layer_cross_v: np.ndarray,
        offset: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
                self.decoder.get_outputs()[1].name,
                self.decoder.get_outputs()[2].name,
            ],
            {
                self.decoder.get_inputs()[0].name: tokens,
                self.decoder.get_inputs()[1].name: n_layer_self_k_cache,
                self.decoder.get_inputs()[2].name: n_layer_self_v_cache,
                self.decoder.get_inputs()[3].name: n_layer_cross_k,
                self.decoder.get_inputs()[4].name: n_layer_cross_v,
                self.decoder.get_inputs()[5].name: offset,
            },
        )
        return logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache


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


def compute_features(filename: str, dim: int = 80) -> np.ndarray:
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
        features.append(f)

    features = np.stack(features, axis=0)

    return features


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def main():
    wave_filename = "./0-zh-en.wav"
    features = compute_features(wave_filename)

    m = OnnxModel(
        encoder="./onnx/encoder.int8.onnx",
        decoder="./onnx/decoder.int8.onnx",
    )
    features = (features - m.cmvn_mean) * m.cmvn_inv_stddev
    features = np.expand_dims(features, axis=0)

    n_layer_cross_k, n_layer_cross_v = m.run_encoder(features)

    n_layer_self_k_cache, n_layer_self_v_cache = m.get_self_cache()
    tokens = np.array([[m.sos]], dtype=np.int64)
    offset = np.array([0], dtype=np.int64)
    results = []
    for i in range(m.max_len):
        logits, n_layer_self_k_cache, n_layer_self_v_cache = m.run_decoder(
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
        )
        max_token_id = logits.argmax()
        if max_token_id == m.eos:
            break
        results.append(max_token_id)
        tokens[0][0] = max_token_id
        offset += 1

    id2token = load_tokens("./tokens.txt")
    text = "".join([id2token[i] for i in results])
    print(text)

    text = text.replace("▁", " ")
    print(text.strip())


if __name__ == "__main__":
    main()
