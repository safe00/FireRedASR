#!/usr/bin/env python3

from fireredasr.models.fireredasr import FireRedAsr
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch

from pathlib import Path

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


@torch.inference_mode()
def main():
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

    batch_size = 1
    T = 1000
    C = 80
    x = torch.rand(batch_size, T, C)
    x_len = torch.tensor([T], dtype=torch.int64)

    encoder_filename = "onnx/encoder.onnx"
    opset_version = 13

    if not Path(encoder_filename).is_file():
        torch.onnx.export(
            model.model.encoder,
            (x, x_len),
            encoder_filename,
            verbose=False,
            opset_version=opset_version,
            input_names=["x", "x_lens"],
            output_names=["encoder_out", "encoder_out_lens", "enc_mask"],
            dynamic_axes={
                "x": {0: "N", 1: "T"},
                "x_lens": {0: "N"},
                "encoder_out": {0: "N", 1: "T"},
                "encoder_out_lens": {0: "N"},
                "enc_mask": {0: "N", 2: "T"},
            }
            if False
            else {
                "x": {1: "T"},
                "encoder_out": {1: "T"},
                "enc_mask": {2: "T"},
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


if __name__ == "__main__":
    main()
