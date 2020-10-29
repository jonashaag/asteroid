from .. import complex_nn
from ..masknn.recurrent import DCCRMaskNet
from .dcunet import BaseDCUNet


class DCCRNet(BaseDCUNet):  # CHECK-JIT
    """DCCRNet as proposed in [1].

    Args:
        architecture (str): The architecture to use, must be "DCCRN-CL".
        stft_kernel_size (int): STFT frame length to use
        stft_stride (int, optional): STFT hop length to use.
        sample_rate (float): Sampling rate of the model.

    References
        - [1] : "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement",
        Yanxin Hu et al. https://arxiv.org/abs/2008.00264
    """

    masknet_class = DCCRMaskNet

    def __init__(
        self, *args, stft_kernel_size=512, sample_rate=16000.0, masknet_kwargs=None, **kwargs
    ):
        super().__init__(
            *args,
            stft_kernel_size=stft_kernel_size,
            sample_rate=sample_rate,
            masknet_kwargs={"n_freqs": stft_kernel_size // 2, **(masknet_kwargs or {})},
            **kwargs,
        )

    def postprocess_encoded(self, tf_rep):
        return complex_nn.as_torch_complex(tf_rep)[..., :-1, :]

    def postprocess_masked(self, masked_tf_rep):
        # pad: [0, 0, ..., 0, 1, 0, 0]
        pad = torch.zeros(2 * masked_tf_rep.ndim)
        pad[-3] = 1
        return from_torchaudio(torch.view_as_real(torch.pad(masked_tf_rep, pad)))
