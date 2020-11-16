from ._dcunet_architectures import make_unet_encoder_decoder_args

# fmt: off
DCCRN_ARCHITECTURES = {
    "DCCRN-CL": (
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        [
            (  1,  16, (5, 2), (2, 1), (2, 0)),
            ( 16,  32, (5, 2), (2, 1), (2, 0)),
            ( 32,  64, (5, 2), (2, 1), (2, 0)),
            ( 64, 128, (5, 2), (2, 1), (2, 0)),
            #(128, 128, (5, 2), (2, 1), (2, 0)),
            #(128, 128, (5, 2), (2, 1), (2, 0)),
        ],
        # Decoders:
        # (in_chan, out_chan, kernel_size, stride, padding, output_padding)
        [
            #(256, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
            #(256, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
            (256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
            (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
            ( 64,  16, (5, 2), (2, 1), (2, 0), (1, 0)),
            ( 32,   1, (5, 2), (2, 1), (2, 0), (1, 0)),
        ],
    ),
}
