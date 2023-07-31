# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Optional

import torch
import torch.nn as nn

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY
from composer.utils.dist import get_local_rank
import math

try:
    import transformer_engine.pytorch as te
except:
    te = None


class MPTMLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        expansion_ratio: int,
        fc_type: str = 'torch',
        device: Optional[str] = None,
    ):
        super().__init__()
        fc_kwargs = {}
        if fc_type != 'te':
            fc_kwargs['device'] = device
        self.up_proj = FC_CLASS_REGISTRY[fc_type](
            d_model,
            expansion_ratio * d_model,
            **fc_kwargs,
        )
        self.act = nn.GELU(approximate='none')
        self.down_proj = FC_CLASS_REGISTRY[fc_type](
            expansion_ratio * d_model,
            d_model,
            **fc_kwargs,
        )
        self.down_proj._is_residual = True  # type: ignore

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))

class CerebrateMLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        expansion_ratio: int,
        max_step_size:int,
        fc_type: str = 'torch',
        device: Optional[str] = None,
        decay_weight_ma: float = 0.99,
        neuron_keep_probability: float = 1.00,
        neuron_keep_steps: int = 2000,
    ):
        super().__init__()
        fc_kwargs = {}
        if fc_type != 'te':
            fc_kwargs['device'] = device
        self.up_proj = FC_CLASS_REGISTRY[fc_type](
            d_model,
            expansion_ratio * d_model,
            **fc_kwargs,
        )
        self.act = nn.GELU(approximate='none')
        self.down_proj = FC_CLASS_REGISTRY[fc_type](
            expansion_ratio * d_model,
            d_model,
            **fc_kwargs,
        )
        self.down_proj._is_residual = True  # type: ignore

        self.iteration = 0
        self.free_neuron_p = 1 / (expansion_ratio * d_model)
        neuron_keep_probability_func = lambda step: \
            neuron_keep_probability + neuron_keep_probability / math.exp(5 * step / max_step_size)
        p_tohold = 1 - neuron_keep_probability_func(neuron_keep_steps)
        self.neuron_keep_probability_func = lambda step: \
            min(1, p_tohold - (p_tohold * max(0,
            (step-neuron_keep_steps)/(max_step_size-neuron_keep_steps)))+\
            neuron_keep_probability + neuron_keep_probability / math.exp(5 * step / max_step_size))
        local_rank = get_local_rank()
        self.neuron_activation = torch.zeros(expansion_ratio * d_model, device=local_rank)
        self.neuron_mask = torch.ones(expansion_ratio * d_model,  device=local_rank)
        self.decay_weight_ma = decay_weight_ma

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)
        mean_activations = torch.mean(torch.mean(torch.abs(x), 0), 0)
        #self.neuron_activation = self.neuron_activation.to(x.device)
        #self.neuron_activation = self.neuron_activation.to(mean_activations.device)
        #neuron_mask = self.neuron_mask.to(x.device)
        self.neuron_activation = (self.decay_weight_ma  * self.neuron_activation) + ((1-self.decay_weight_ma) * mean_activations)
        #self.neuron_activation = self.neuron_activation.to('cpu')
        #self.neuron_activation = self.neuron_activation.to('meta')
        keep_neuron_p = self.neuron_keep_probability_func(self.iteration)
        neuron_available_p = torch.sum(self.neuron_mask) / self.neuron_mask.size(dim=0)

        if keep_neuron_p < (neuron_available_p - self.free_neuron_p):
            num_neurons_to_kill = (neuron_available_p - keep_neuron_p) // self.free_neuron_p
            num_neurons_to_kill = num_neurons_to_kill.cpu().numpy()
            num_neurons_to_kill = num_neurons_to_kill.astype('int')
            num_neurons_to_kill = int(num_neurons_to_kill)
            if num_neurons_to_kill > 0:
                neuron_activations_active = torch.mul(self.neuron_activation, self.neuron_mask)
                maximum_value_temp = torch.max(neuron_activations_active) + 1
                neuron_activations_active[neuron_activations_active==0] = maximum_value_temp
                values, indices = torch.topk(neuron_activations_active, num_neurons_to_kill, largest=False)
                self.neuron_mask[indices] = 0
                #neuron_mask[indices] = 0
        x = torch.mul(x, self.neuron_mask.view(1, 1, -1))

        x = self.down_proj(x)

        self.iteration += 1

        return x

FFN_CLASS_REGISTRY = {
    'mptmlp': MPTMLP,
    'cerebrate_mlp': CerebrateMLP,
}


def build_ffn(
    d_model: int,
    expansion_ratio: int,
    max_step_size: int = 1000,
    fc_type: str = 'torch',
    device: Optional[str] = None,
    decay_weight_ma: float = 0.99,
    neuron_keep_probability: float = 1.00,
    neuron_keep_steps: int = 2000,
    **kwargs,
):
    ffn_type = 'cerebrate_mlp'
    if ffn_type == 'mptmlp':
        if kwargs is not None and len(kwargs) > 0:
            raise ValueError(
                f'MPTMLP got an unexpected keyword argument: {kwargs}')
        return MPTMLP(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            fc_type=fc_type,
            device=device,
        )
    elif ffn_type == 'te_ln_mlp':
        return te.LayerNormMLP(
            hidden_size=d_model,
            ffn_hidden_size=d_model * expansion_ratio,
            **kwargs,
        )
    elif ffn_type == 'cerebrate_mlp':
        return CerebrateMLP(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            fc_type=fc_type,
            device=device,
            decay_weight_ma=decay_weight_ma,
            neuron_keep_probability=neuron_keep_probability,
            neuron_keep_steps=neuron_keep_steps,
            max_step_size=max_step_size,
        )

    else:
        raise ValueError(f'{ffn_type=} not recognized.')
