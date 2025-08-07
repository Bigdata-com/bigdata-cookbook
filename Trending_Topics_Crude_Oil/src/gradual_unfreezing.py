import json
import math

from typing import Optional
from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    Trainer,
    )
from transformers import PreTrainedModel
from torch import nn

def extract_model_info(model: PreTrainedModel):
    num_encoder_layers = model.config.num_hidden_layers
    model_children = [name for name, _ in model.named_children()]
    main_module_name = model_children.pop(0)
    main_module = getattr(model, main_module_name)
    main_module_children = [name for name, _ in main_module.named_children()]
    assert 'embeddings' == main_module_children.pop(0), 'Embeddings not the first module in main module.'
    encoder_name = main_module_children.pop(0)
    assert encoder_name in ['transformer', 'encoder'], 'Encoder not the second module!'
    return main_module, encoder_name, num_encoder_layers

def get_module(main_module: nn.Module, module_name: str, layer: Optional[int] = None):
    if module_name == 'embeddings':
        return getattr(main_module, 'embeddings')
    elif module_name in ['transformer', 'encoder']:
        return getattr(main_module, module_name).layer[layer]
    else:
        raise ValueError(
            f"Unexpected module name '{module_name}' provided. "
            "Expected 'embeddings', 'transformer', or 'encoder'.")

def get_module_by_layer(model: PreTrainedModel, layer: int):
    main_module, encoder_name, total_layers = extract_model_info(model)
    if layer == 0:
        module = get_module(main_module, 'embeddings')
    elif 0 < layer <= total_layers + 1:
        module = get_module(main_module, encoder_name, layer-1)
    else:
        raise ValueError(f"Layer number should be between 0 and {total_layers}")
    return module

def freeze_all_layers(model: PreTrainedModel):
    _, _, total_num_layers = extract_model_info(model)
    for layer_id in range(total_num_layers + 1):
        layer_module = get_module_by_layer(model, layer_id)
        freeze_module(layer_module)


def freeze_module(module: nn.Module):
    for params in module.parameters():
        params.requires_grad = False

def calculate_checksum(main_module: nn.Module, module_name: str, layer: Optional[int] = None):
    layer_module = get_module(main_module, module_name, layer)
    return sum([params.sum().item() for params in layer_module.parameters()])

def get_learnable_status(model: PreTrainedModel):
    main_module, encoder_name, total_num_layers = extract_model_info(model)
    status = {'embeddings': is_learnable(main_module, 'embeddings')}
    for i in range(total_num_layers):
        status[f'encoder_{i}'] = is_learnable(main_module, encoder_name, i)
    return status

def get_checksums(model: PreTrainedModel):
    main_module, encoder_layer_name, total_num_layers = extract_model_info(model)
    checksum = {'embeddings': calculate_checksum(main_module, 'embeddings')}
    for i in range(total_num_layers):
        checksum[f'encoder_{i}'] = calculate_checksum(main_module, encoder_layer_name, i)
    return checksum

def is_learnable(main_module: nn.Module, layer_name: str, layer: Optional[int] = None):
    layer_module = get_module(main_module, layer_name, layer)
    return any(param.requires_grad for param in layer_module.parameters())

def unfreeze_layer_by_id(model: PreTrainedModel, layer: int):
    layer_module = get_module_by_layer(model, layer)
    unfreeze_layer(layer_module)

def unfreeze_layer(layer_module: nn.Module):
    for params in layer_module.parameters():
        params.requires_grad = True

def print_learnable_params(model: PreTrainedModel):
    for name, param in model.named_parameters():
        print(param.requires_grad, name)

class RavenFreezingCallback(TrainerCallback):
    def __init__(
            self,
            num_layers_to_unfreeze: int,
            debug: bool = False
            ) -> None:
        self.debug = debug
        self.num_layers_to_unfreeze: int = num_layers_to_unfreeze  # Include embeddings
        self.epoch = 0
        self.step = 0
        self.unfrozen_layers = 1  # Initially, only the classification layer will be unfrozen.

        self.status_learnable: dict
        self.status_checksum: dict
        self.steps_per_epoch: int
        self.steps_per_training: int
        self.steps_freeze_duration: float
        self.total_num_layers: int


    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        # for param in model.bert.parameters():
        #     param.requires_grad = False
        freeze_all_layers(model)
        if self.debug:
            self.status_learnable = get_learnable_status(model)
            self.status_checksum = get_checksums(model)
            print('What parts of the encoder will be trained at the start:', get_learnable_status(model))
            # print_learnable_params(model)
            print('End of Trainer initialization observed in RavenFreezer.')
        _, _, self.total_num_layers = extract_model_info(model)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        self.steps_per_epoch = len(kwargs['train_dataloader'])/args.gradient_accumulation_steps
        self.steps_per_training = state.max_steps
        self.steps_freeze_duration = self.steps_per_training / (self.num_layers_to_unfreeze + 1)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        self.epoch += 1
        # print(f'------------->Epoch: {self.epoch}; Step: {self.step};')

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        self.step += 1
        if self.steps_freeze_duration is None:
            raise ValueError("Unknow duration for each freeze.")

        if (self.step / self.steps_freeze_duration) > self.unfrozen_layers:
            layers_to_train = math.ceil(self.step / self.steps_freeze_duration)
            layers_to_unfreeze = layers_to_train - self.unfrozen_layers
            list_of_layers_to_unfreeze = list(range(self.total_num_layers-(self.unfrozen_layers-1), self.total_num_layers-(self.unfrozen_layers-1)-layers_to_unfreeze, -1))
            if self.debug:
                print('We will unfreeze additional layers')
                print(f'Currently we have {self.unfrozen_layers} unfozen layers')
                print(f'The number of layers we want to be training from this step onwards: {layers_to_train}')
                print(f'We will unfreeze an additional {layers_to_unfreeze} layers.')
                if layers_to_unfreeze > 1:
                    print('Unfreezing more than 2 layers simultaneously')
                # print_learnable_params(model)
                new_checksums = get_checksums(model)
                changes = {layer: ['Changed', '-'][val == self.status_checksum[layer]] for layer, val in new_checksums.items()}
                learnable_status = get_learnable_status(model)
                print(json.dumps({layer: [changes[layer], ['-', 'Was learned this time'][learnable_status[layer]]] for layer in changes}, indent=4))
                self.status_checksum = new_checksums
                print('\n')
            for new_layer_to_unfreeze in list_of_layers_to_unfreeze:
                if new_layer_to_unfreeze >= 0:  # Only unfreeze if layer index is valid
                    unfreeze_layer_by_id(model, new_layer_to_unfreeze)
            self.unfrozen_layers += layers_to_unfreeze

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        new_checksums = get_checksums(model)
        print({layer: ['Changed', 'Did not change'][val == self.status_checksum[layer]] for layer, val in new_checksums.items()})
