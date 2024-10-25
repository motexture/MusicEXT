# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
AudioCraft is a general framework for training audio generative models.
At the moment we provide the training code for:

- [MusicGen](https://arxiv.org/abs/2306.05284), a state-of-the-art
    text-to-music and melody+text autoregressive generative model.
    For the solver, see `musicext.solvers.musicgen.MusicGenSolver`, and for the model,
    `musicext.models.musicgen.MusicGen`.
- [AudioGen](https://arxiv.org/abs/2209.15352), a state-of-the-art
    text-to-general-audio generative model.
- [EnCodec](https://arxiv.org/abs/2210.13438), efficient and high fidelity
    neural audio codec which provides an excellent tokenizer for autoregressive language models.
    See `musicext.solvers.compression.CompressionSolver`, and `musicext.models.encodec.EncodecModel`.
- [MultiBandDiffusion](TODO), alternative diffusion-based decoder compatible with EnCodec that
    improves the perceived quality and reduces the artifacts coming from adversarial decoders.
"""

# flake8: noqa
from . import data, modules, models

__version__ = '1.4.0a1'
