# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Optional

import torch
import torch.nn as nn

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY
from composer.utils import dist
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
        self.number_of_neurons = expansion_ratio * d_model
        self.free_neuron_p = 1 / (expansion_ratio * d_model)
        neuron_keep_probability_func = lambda step: \
            neuron_keep_probability + (1 - neuron_keep_probability) / math.exp(5 * step / max_step_size)
        p_tohold = 1 - neuron_keep_probability_func(neuron_keep_steps)
        self.neuron_keep_probability_func = lambda step: \
            min(1, p_tohold - (p_tohold * max(0,
            (step-neuron_keep_steps)/(max_step_size-neuron_keep_steps)))+\
            neuron_keep_probability + (1 - neuron_keep_probability) / math.exp(5 * step / max_step_size))
        self.neuron_activation = torch.zeros(expansion_ratio * d_model, device=device)
        self.neuron_mask = torch.ones(expansion_ratio * d_model,  device=device)
        self.decay_weight_ma = decay_weight_ma
        self.max_step_size = max_step_size
        self.neuron_keep_probability = neuron_keep_probability
        self.neuron_keep_steps = neuron_keep_steps

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)

        neuron_mask = self.neuron_mask.to(x.device)
        if self.training:
            mean_activations = torch.mean(torch.mean(torch.abs(x), 0), 0)
            #self.neuron_activation = self.neuron_activation.to(x.device)
            #self.neuron_activation = self.neuron_activation.to(mean_activations.device)
            #neuron_mask = self.neuron_mask.to(x.device)
            #neuron_activation = dist.all_gather(self.neuron_activation)

            neuron_activation = self.neuron_activation.to(x.device)

            #self.neuron_activation = (self.decay_weight_ma  * self.neuron_activation) + ((1-self.decay_weight_ma) * mean_activations)
            neuron_activation *= self.decay_weight_ma
            neuron_activation +=  ((1-self.decay_weight_ma) * mean_activations)

            if self.iteration % 100 == 0:
                neuron_activation_sorted, indices = torch.sort(neuron_activation)
                neuron_activation_max = neuron_activation_sorted[-1]
                neuron_activation_90 = neuron_activation_sorted[self.number_of_neurons*9//10]
                neuron_activation_75 = neuron_activation_sorted[self.number_of_neurons*3//4]
                neuron_activation_median = neuron_activation_sorted[self.number_of_neurons//2]
                neuron_activation_25 = neuron_activation_sorted[self.number_of_neurons//4]
                neuron_activation_10 = neuron_activation_sorted[self.number_of_neurons//10]
                neuron_activation_min = neuron_activation_sorted[0]
                print(f'Neuron activation values in iteration {self.iteration}:\nmin: {neuron_activation_min}\n'
                      f'10_p: {neuron_activation_10}\n'
                      f'25_p: {neuron_activation_25}\n'
                      f'median: {neuron_activation_median}\n'
                      f'75_p: {neuron_activation_75}\n'
                      f'90_p: {neuron_activation_90}\n'
                      f'max: {neuron_activation_max}')

            debug = False
            self.neuron_activation = neuron_activation.detach().cpu()
            #self.neuron_activation = self.neuron_activation.to('cpu')
            #self.neuron_activation = self.neuron_activation.to('meta')
            keep_neuron_p = self.neuron_keep_probability_func(self.iteration)
            neuron_available_p = torch.sum(neuron_mask) / neuron_mask.size(dim=0)

            if keep_neuron_p < (neuron_available_p - self.free_neuron_p):
                if debug:
                    print('############################')
                    print(f'iteration: {self.iteration}')
                    print(f'keep_neuron_p: {keep_neuron_p}')
                    print(f'neuron_available_p: {neuron_available_p}')
                    print(f'self.free_neuron_p: {self.free_neuron_p}')
                    print(f'neuron_keep_probability: {self.neuron_keep_probability}')
                    print(f'neuron_keep_steps: {self.neuron_keep_steps}')
                    print(f'max_step_size: {self.max_step_size}')
                    print('###################################################################')
                    print('Somethinggg wrongg')
                    print('###################################################################')
                num_neurons_to_kill = (neuron_available_p - keep_neuron_p) // self.free_neuron_p
                num_neurons_to_kill = num_neurons_to_kill.cpu().numpy()
                num_neurons_to_kill = num_neurons_to_kill.astype('int')
                num_neurons_to_kill = int(num_neurons_to_kill)
                if num_neurons_to_kill > 0:
                    neuron_activations_active = torch.mul(neuron_activation, neuron_mask)
                    neuron_activations_active = neuron_activations_active.detach().cpu()
                    maximum_value_temp = torch.max(neuron_activations_active) + 1
                    neuron_activations_active[neuron_activations_active==0] = maximum_value_temp
                    values, indices = torch.topk(neuron_activations_active, num_neurons_to_kill, largest=False)
                    self.neuron_mask[indices] = 0
                    neuron_mask[indices] = 0
                    del neuron_activations_active, maximum_value_temp, values, indices
                del num_neurons_to_kill

            del neuron_activation,  neuron_available_p, keep_neuron_p, mean_activations
            torch.cuda.empty_cache()

            self.iteration += 1
        #x = torch.mul(x, neuron_mask.view(1, 1, -1))
        del neuron_mask
        x = self.down_proj(x)

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
