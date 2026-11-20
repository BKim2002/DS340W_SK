"""LowRankLinear copied from the project's patched original OpenDelta source.

Upstream: thunlp/OpenDelta, commit 20e426f716ea53618657cd34bbfc4b347b98cd74,
opendelta/delta_models/lora.py. The optional scaling-mode change is the same
patches/0006-optional-lora-scaling.patch used for the retained experiments.
The original LoRA implementation and copyright notice are preserved below.
See licenses/OpenDelta-APACHE-2.0.txt and licenses/LoRA-MIT.txt.
"""
import math
import torch.nn as nn

class LowRankLinear(nn.Module):
    #  ------------------------------------------------------------------------------------------
    #  Copyright (c) Microsoft Corporation. All rights reserved.
    #  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
    #  ------------------------------------------------------------------------------------------
    #  copy from loralib and do some refactor
    def __init__(self,
        in_features,
        out_features,
        weight,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_scaling_mode='original',
    ):
        super().__init__()
        if lora_scaling_mode not in ('original', 'sqrt_rank'):
            raise ValueError('Unknown LoRA scaling mode: ' + str(lora_scaling_mode))
        self.lora_scaling_mode = lora_scaling_mode
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        if r > 0:
            self.lora_A = nn.Parameter(weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / (math.sqrt(self.r) if self.lora_scaling_mode == 'sqrt_rank' else self.r)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
