from ._dcunet_architectures import make_unet_encoder_decoder_args

# fmt: off
DCCRN_ARCHITECTURES = {
    "DCCRN-CL": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding)
        [
            (  1,  16, (5, 2), (2, 1), (2, 0)),
            ( 16,  32, (5, 2), (2, 1), (2, 1)),
            ( 32,  64, (5, 2), (2, 1), (2, 0)),
            ( 64, 128, (5, 2), (2, 1), (2, 1)),
            (128, 128, (5, 2), (2, 1), (2, 0)),
            (128, 128, (5, 2), (2, 1), (2, 1)),
        ],
        # Decoders: auto
        "auto",
        # First decoder has a skip connection from the intermediate layer (LSTM) output.
        first_decoder_has_concatenative_skip_connection=True,
    ),
}