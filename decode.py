#!/usr/bin/env python3

from pathlib import Path

import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor

from fireredasr.data.asr_feat import CMVN, ASRFeatExtractor
from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.models.module.transformer_decoder import (
    DecoderLayer,
    DecoderMultiHeadAttention,
    TransformerDecoder,
)

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
import onnxruntime
import soundfile as sf


class ModifiedEncoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, x: torch.Tensor, x_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (N, T, C)
          x_len: (N,)
        Returns:
          a tuple containing:
            - n_layer_cross_k_cache: (num_dec_layers, N, T, d_model)
            - n_layer_cross_v_cache: (num_dec_layers, N, T, d_model)
        """
        encoder_out, _, _ = self.encoder(x, x_len)

        n_layer_cross_k_list = []
        n_layer_cross_v_list = []
        for layer in self.decoder.layer_stack:
            k = layer.cross_attn.w_ks(encoder_out)
            v = layer.cross_attn.w_vs(encoder_out)
            n_layer_cross_k_list.append(k)
            n_layer_cross_v_list.append(v)

        return torch.stack(n_layer_cross_k_list), torch.stack(n_layer_cross_v_list)


class MultiHeadAttentionSelf(torch.nn.Module):
    def __init__(self, attn: DecoderMultiHeadAttention):
        super().__init__()
        self.attn = attn

    def forward(
        self,
        x: Tensor,  # (b, n_ctx      , n_state)
        k_cache: Tensor,  # (b, n_ctx_cache, n_head, d_k)
        v_cache: Tensor,  # (b, n_ctx_cache, n_head, d_k)
    ):
        bs = x.size(0)

        q = self.attn.w_qs(x).view(bs, -1, self.attn.n_head, self.attn.d_k)
        k = self.attn.w_ks(x).view(bs, -1, self.attn.n_head, self.attn.d_k)
        v = self.attn.w_vs(x).view(bs, -1, self.attn.n_head, self.attn.d_k)

        k_cache[:, -k.shape[1] :, :] = k  # (b, n_ctx_cache + n_ctx, n_head, d_k)
        v_cache[:, -v.shape[1] :, :] = v  # (b, n_ctx_cache + n_ctx, n_head, d_k)

        q = q.transpose(1, 2)

        output = self.attn.attention(
            q,
            k_cache.transpose(1, 2),
            v_cache.transpose(1, 2),
        )

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.attn.d_model)

        output = self.attn.fc(output)

        return output, k_cache, v_cache


class MultiHeadAttentionCross(torch.nn.Module):
    def __init__(self, attn: DecoderMultiHeadAttention):
        super().__init__()
        self.attn = attn

    def forward(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
    ):
        bs = x.size(0)

        q = self.attn.w_qs(x).view(bs, -1, self.attn.n_head, self.attn.d_k)
        k = k.view(bs, -1, self.attn.n_head, self.attn.d_k)
        v = v.view(bs, -1, self.attn.n_head, self.attn.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = self.attn.attention(q, k, v)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.attn.d_model)
        output = self.attn.fc(output)

        return output


class ModifiedDecoderLayer(torch.nn.Module):
    def __init__(self, layer: DecoderLayer):
        super().__init__()
        self.layer = layer
        self.attn = MultiHeadAttentionSelf(layer.self_attn)
        self.cross_attn = MultiHeadAttentionCross(layer.cross_attn)

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
    ):
        """
        Args:
          x: (N, 1, d)
          self_k_cache: (N, num_processed_tokens, num_head, head_dim)
          self_v_cache: (N, num_processed_tokens, num_head, head_dim)
          cross_k: (N, T, d)
          cross_v: (N, T, d)
        """
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(
            self.layer.self_attn_norm(x),
            self_k_cache,
            self_v_cache,
        )

        x = x + self_attn_x

        x = x + self.cross_attn(self.layer.cross_attn_norm(x), cross_k, cross_v)

        x = x + self.layer.mlp(self.layer.mlp_norm(x))

        return x, self_k_cache_updated, self_v_cache_updated


class ModifiedTransformerDecoder(torch.nn.Module):
    def __init__(self, decoder: TransformerDecoder, max_len=1024):
        super().__init__()
        self.decoder = decoder

        self.blocks = []
        for orginal_block in self.decoder.layer_stack:
            self.blocks.append(ModifiedDecoderLayer(orginal_block))

    def forward(
        self,
        tokens: Tensor,
        n_layer_self_k_cache: Tensor,
        n_layer_self_v_cache: Tensor,
        n_layer_cross_k: Tensor,
        n_layer_cross_v: Tensor,
        offset: Tensor,
    ):
        """
        Args:
          tokens: (N, num_tokens)
        """
        x = (
            self.decoder.tgt_word_emb(tokens) * self.decoder.scale
            + self.decoder.positional_encoding.pe[
                :, offset[0] : offset[0] + tokens.shape[-1]
            ]
        )

        x = x.to(n_layer_cross_k[0].dtype)

        i = 0
        for block in self.blocks:
            self_k_cache = n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :]
            self_v_cache = n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :]

            x, self_k_cache, self_v_cache = block(
                x,
                self_k_cache=self_k_cache,
                self_v_cache=self_v_cache,
                cross_k=n_layer_cross_k[i],
                cross_v=n_layer_cross_v[i],
            )

            n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_k_cache
            n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_v_cache
            i += 1

        x = self.decoder.layer_norm_out(x)

        logits = self.decoder.tgt_word_prj(x)
        return logits, n_layer_self_k_cache, n_layer_self_v_cache


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

    x = x.unsqueeze(0)

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

    encoder = ModifiedEncoder(model.model.encoder, model.model.decoder)
    x_len = torch.tensor([x.shape[1]], dtype=torch.int64)

    n_layer_cross_k, n_layer_cross_v = encoder(x, x_len)
    print(n_layer_cross_k.shape)  # (16, 1, 166, 1280)
    print(n_layer_cross_v.shape)
    assert (
        n_layer_cross_k.shape[0] == model.model.decoder.n_layers
    ), n_layer_cross_k.shape
    assert n_layer_cross_k.shape[1] == 1, "batch size is 1"
    # n_layer_cross_k_cache.shape[2] is T

    d_model = model.model.decoder.tgt_word_prj.weight.shape[1]

    assert n_layer_cross_k.shape[3] == d_model, "batch size is 1"

    max_len = 1024

    decoder = ModifiedTransformerDecoder(model.model.decoder, max_len)
    num_head = decoder.blocks[0].attn.attn.n_head
    head_dim = decoder.blocks[0].attn.attn.d_k
    n_layer_self_k_cache = torch.zeros(
        (
            len(model.model.decoder.layer_stack),
            1,  # batch size
            max_len,
            num_head,
            head_dim,
        ),
    )
    n_layer_self_v_cache = torch.zeros(
        (len(model.model.decoder.layer_stack), 1, max_len, num_head, head_dim),
    )
    offset = torch.zeros(1, dtype=torch.int64)

    tokens = torch.tensor([[model.model.decoder.sos_id]])

    print("eos id", model.model.decoder.eos_id)
    results = []
    for i in range(0, max_len):
        logits, n_layer_self_k_cache, n_layer_self_v_cache = decoder(
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
        )
        max_token_id = logits.argmax(dim=-1)
        if max_token_id.item() == model.model.decoder.eos_id:
            break
        results.append(max_token_id.item())
        tokens = max_token_id
        offset += 1
    print(results)
    text = model.tokenizer.detokenize(results)
    print(text)

    opset_version = 13
    decoder_filename = f"onnx/decoder.onnx"

    offset = torch.zeros(1, dtype=torch.int64)

    if not Path(decoder_filename).is_file():
        torch.onnx.export(
            decoder,
            (
                tokens,
                n_layer_self_k_cache,
                n_layer_self_v_cache,
                n_layer_cross_k,
                n_layer_cross_v,
                offset,
            ),
            decoder_filename,
            opset_version=opset_version,
            input_names=[
                "tokens",
                "in_n_layer_self_k_cache",
                "in_n_layer_self_v_cache",
                "n_layer_cross_k",
                "n_layer_cross_v",
                "offset",
            ],
            output_names=[
                "logits",
                "out_n_layer_self_k_cache",
                "out_n_layer_self_v_cache",
            ],
            dynamic_axes={
                "in_n_layer_self_k_cache": {2: "T"},
                "in_n_layer_self_v_cache": {2: "T"},
                "n_layer_cross_k": {2: "T"},
                "n_layer_cross_v": {2: "T"},
            },
        )

    decoder_filename_int8 = "onnx/decoder.int8.onnx"
    if not Path(decoder_filename_int8).is_file():
        quantize_dynamic(
            model_input=decoder_filename,
            model_output=decoder_filename_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )

    return

    encoder_filename = "onnx/encoder.onnx"

    if not Path(encoder_filename).is_file():
        x0 = torch.rand(1, 1000, 80)
        x0_len = torch.tensor([x0.shape[1]], dtype=torch.int64)
        torch.onnx.export(
            encoder,
            (x0, x0_len),
            encoder_filename,
            verbose=False,
            opset_version=opset_version,
            input_names=["x"],
            output_names=["n_layer_cross_k", "n_layer_cross_v"],
            dynamic_axes={
                "x": {0: "N", 1: "T"},
                "n_layer_cross_k": {1: "N", 2: "T"},
                "n_layer_cross_v": {1: "N", 2: "T"},
            }
            if False
            else {
                "x": {1: "T"},
                "n_layer_cross_k": {2: "T"},
                "n_layer_cross_v": {2: "T"},
            },
        )

    encoder_filename_int8 = "onnx/encoder.int8.onnx"
    if not Path(encoder_filename_int8).is_file():
        quantize_dynamic(
            model_input=encoder_filename,
            model_output=encoder_filename_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )

    return

    onnx = OnnxModel(encoder="./onnx/encoder.int8.onnx")

    x = x.unsqueeze(0)
    use_onnx = True

    if use_onnx:
        enc_outputs, _, enc_mask = onnx.run_encoder(x)
        model.model.decoder(enc_outputs)
        return
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
