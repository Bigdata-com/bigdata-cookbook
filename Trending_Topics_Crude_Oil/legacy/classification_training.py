import argparse
from collections import Counter
from hashlib import blake2b
import json
import os
import shutil
from typing import Any, Literal, Optional, Union
from collections import Counter
import random 
import gc

from datasets import Dataset, DatasetDict, load_from_disk
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import EvalPrediction, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import Trainer, TrainingArguments, TrainerCallback, PreTrainedModel, EarlyStoppingCallback


try:
    from tools.gradual_unfreezing import RavenFreezingCallback
except ModuleNotFoundError:
    from gradual_unfreezing import RavenFreezingCallback


SIGMA = 0.4

DEFAULT_TRAINING_ARGS = {
    "evaluation_strategy": "epoch",
    "per_device_train_batch_size": 4, # No of sentences fed to the model.. for training
    "per_device_eval_batch_size": 4,  # ..For eval
    "gradient_accumulation_steps": 1, # How many steps to accummulate the gradient -> modify per batches
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "num_train_epochs": 5,
    "warmup_ratio": 0.1,
    "save_strategy": "epoch",
    "logging_strategy": "epoch", #or "steps"
    "logging_steps": 200, # only used if logging_strategy  = 'steps'
    "save_only_model": True,
    "save_total_limit": 1,
    "seed": 40,
    "fp16": torch.cuda.is_available(),
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",  # "f1_macro",  
    "greater_is_better" : False, #best model selection for greater/lower score? change to true for f1score for example
    # "label_smoothing_factor": 0.1,  # may use this for uncertain labels.
    "report_to": ["tensorboard"],
    "logging_first_step": True
    # "disable_tqdm":True
}


def gaussian_labels(labels: torch.Tensor, num_classes: int, sigma: float = SIGMA):
    """
    Returns Gaussian distributed labels around the true labels.

    This function takes a tensor of class labels and the number of classes, and converts the labels into a Gaussian distribution. The sigma parameter controls the width of the Gaussian distribution.

    Args:
        labels (torch.Tensor): A tensor of class labels.
        num_classes (int): The number of classes.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to SIGMA.

    Returns:
        torch.Tensor: A tensor of the same shape as labels, but with the labels converted into a Gaussian distribution.
    """
    base = torch.arange(0, num_classes).float().to(labels.device)
    labels = labels.float().unsqueeze(-1).to(labels.device)
    labels_gaussian = torch.exp(-0.5 * (labels - base) ** 2 / sigma ** 2)
    labels_gaussian /= labels_gaussian.sum(dim=-1, keepdim=True)
    return labels_gaussian


class RavenEarlyStoppingCallback(EarlyStoppingCallback):
    """
    A callback for early stopping with verbose logging.

    This class extends the EarlyStoppingCallback to include verbose logging of the training progress.

    Args:
        early_stopping_patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 1.
        early_stopping_threshold (float, optional): The improvement over the best loss needed to count as a better model. Defaults to 0.0.
        verbose (bool, optional): If True, prints detailed logging information. Defaults to False.
    """
    def __init__(self, early_stopping_patience: int = 1,
                 early_stopping_threshold: Optional[float] = 0.0,
                 verbose: bool = False):
        self.verbose = verbose
        super().__init__(early_stopping_patience, early_stopping_threshold)

    def check_metric_value(self, args, state, control, metric_value):
        counter = self.early_stopping_patience_counter
        super().check_metric_value(args, state, control, metric_value)
        if self.verbose:
            if counter == self.early_stopping_patience_counter:
                print(f'Step: {state.global_step}; '
                      f'Epoch: {state.epoch:.2f}; '
                      f'Metric: {metric_value:.6f}; '
                      f'Best metric {state.best_metric or 0:.6f}; '
                      f'Counter: {counter}')
            else:
                print(f'Step: {state.global_step}; '
                      f'Epoch: {state.epoch:.2f}; '
                      f'Metric: {metric_value:.6f}; '
                      f'Best metric {state.best_metric or 0:.6f}; '
                      f'Counter: {counter} -> {self.early_stopping_patience_counter}')


    def get_logits(example, model, tokenizer):
        inputs = tokenizer(example['_text_column'], return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        return {'_logits': logits.cpu().numpy()}
        
def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Get the latest checkpoint directory from the output directory.

    Parameters:
    output_dir (str): The directory where checkpoints are stored.

    Returns:
    Optional[str]: The name of the latest checkpoint directory, or None if no valid checkpoint directory is found.
    """

    if not os.path.isdir(output_dir):
        return None

    checkpoint_dir = [folder for folder in os.listdir(output_dir) if 'checkpoint' in folder]
    checkpoint_dir = checkpoint_dir[0] if len(checkpoint_dir) == 1 else None

    return checkpoint_dir


def get_best_model_checkpoint(output_dir: str) -> Union[None, str]:
    """
    Get the best model checkpoint from the output directory.

    Parameters:
    output_dir (str): The directory where checkpoints are stored.

    Returns:
    Union[None, str]: The path to the best model checkpoint, or None if no valid checkpoint is found.
    """

    ckpt_dirs = [ckpt_dir for ckpt_dir in os.listdir(output_dir)
                if ('checkpoint' in ckpt_dir) and os.path.isdir(os.path.join(output_dir, ckpt_dir))]

    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))

    if len(ckpt_dirs) > 0:
        last_ckpt = ckpt_dirs[-1]
    else:
        return None

    state = TrainerState.load_from_json(
        os.path.join(output_dir, last_ckpt, 'trainer_state.json'))

    return state.best_model_checkpoint
    

def get_logits(
    example: str,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    ):
    """
    Computes the logits for a given text example using a sequence classification model.

    This function tokenizes the input text, feeds it into the model, and returns the computed logits.

    Args:
        example (str): The text example to compute logits for.
        model (AutoModelForSequenceClassification): The sequence classification model to use.
        tokenizer (AutoTokenizer): The tokenizer to use for the text example.

    Returns:
        dict: A dictionary with the computed logits under the key '_logits'.
    """
    
    device = model.device
    inputs = tokenizer(example['_text_column'], return_tensors='pt', padding=True)

    max_l = min(model.config.max_position_embeddings, tokenizer.model_max_length)

    cond = ([len(i) > max_l for i in inputs['input_ids']] )

    inputs = tokenizer(example['_text_column'], return_tensors='pt', padding=True,  truncation=True)


    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits

    logits = logits.cpu().numpy()
    
    logits[cond]= np.nan
    
    return {'_logits': logits}

def evaluate_performance(
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dataset: Dataset,
        text_column: str,
        device: Optional[str] = None,
        return_inferred_labels: bool = False,
        batch_size: int = 8  # Default batch size
        ):
    """
    Evaluates the performance of a sequence classification model on a given dataset.

    This function computes the logits for each example in the dataset, and then computes the performance metrics based on the problem type.

    Args:
        model (AutoModelForSequenceClassification): The sequence classification model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer to use for the text examples.
        dataset (Dataset): The dataset to evaluate the model on.
        text_column (str): The name of the column in the dataset that contains the text examples.
        device (Optional[str]): The device to run the evaluation on. If None, defaults to 'cuda' if available, else 'cpu'.
        return_inferred_labels (bool): Whether to return inferred labels along with performance metrics. Defaults to False.
        batch_size (int): The batch size to use for evaluation. Defaults to 8.
        
    Returns:
        dict[str, float]: A dictionary containing the computed performance metrics.
        DatasetDict: The dataset with the logits for each class and predicted class (if requested)
    """     
    
    _device = (device or torch.device('cuda')) if torch.cuda.is_available() else torch.device('cpu')
    model.to(_device)
    model.eval()

    dataset = dataset.rename_columns({text_column: '_text_column'}).map(
        get_logits,
        fn_kwargs={'model': model, 'tokenizer': tokenizer},
        batched=True,
        batch_size=batch_size, 
        load_from_cache_file=False,
        cache_file_name=None,
        keep_in_memory=True
    )

    dataset = dataset.filter(lambda x: np.isnan(x['_logits']).sum()==0)

    dataset = dataset.with_format('numpy')
    logits = dataset['_logits']
    labels = dataset['labels']

    if hasattr(dataset.features['labels'], '_int2str'):
        dataset_label_order = dataset.features['labels']._int2str
        model_label_mapping = {label.lower(): col_idx for label, col_idx in model.config.label2id.items()}

        model_shares_labels_with_dataset = all(
            [label.lower() in model_label_mapping for label in dataset_label_order])

        if model_shares_labels_with_dataset:
            reordered_logits = np.zeros_like(logits)

            for idx_to, label in enumerate(dataset_label_order):
                idx_from = model_label_mapping[label]
                reordered_logits[:, idx_to] = logits[:, idx_from]
            logits = reordered_logits
    else:
        print('Warning: no information about dataset label order. Proceed with caution if the model was trained with a different type of dataset.')

    if return_inferred_labels:
        pred_probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        for class_id in range(pred_probs.shape[1]):
            dataset = dataset.add_column(
                f'P(class={class_id})', pred_probs[:, class_id])
        predictions = np.argmax(logits, axis=-1)
        dataset = dataset.add_column('predicted_class', predictions)
        return compute_classification_metrics((logits, labels)), dataset
    
    return compute_classification_metrics((logits, labels))


def compute_classification_metrics(
        eval_pred: Union[EvalPrediction, tuple[Any, Any]]) -> dict[str, float]:
    """
    Computes evaluation metrics based on the model predictions and labels.
    
    Args:
        eval_pred (Tuple[np.ndarray, np.ndarray]): A tuple containing the model's logits and labels.

    Returns:
        dict[str, float]: A dictionary of computed evaluation metrics.

    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    mean_absolute_error = np.abs(predictions-labels).mean()

    pred_probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    predictions_weighted = (
        pred_probs * np.arange(pred_probs.shape[1])).sum(axis=1)

    mean_absolute_error_weighted = np.abs(predictions_weighted - labels).mean()

    probs_true = np.zeros_like(pred_probs)
    # idx_true_label = [(k, int(v)) for k, v in enumerate(labels) if not np.isnan(v)]
    probs_true[range(probs_true.shape[0]), labels] = 1
    # probs_true[list(zip(*idx_true_label))] = 1
    mean_cross_entropy_loss = (-np.log(pred_probs) * probs_true).sum(axis=1).mean()

    num_classes = logits.shape[1]
    labels_gaussian = gaussian_labels(torch.tensor(labels), num_classes)
    mean_gaussian_cross_entropy_loss = (-np.log(pred_probs) * labels_gaussian.cpu().numpy()).sum(axis=1).mean()

    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load('f1')
    metric_precision = evaluate.load('precision')
    metric_recall = evaluate.load('recall')

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)

    precisions = {}
    for label in range(num_classes):
        precisions[f'Precision[class={label}]'] = metric_precision.compute(
            predictions=(predictions == label).astype(int),
            references=(labels == label).astype(int),
            zero_division=0
        )['precision']

    recalls = {}
    for label in range(num_classes):
        recalls[f'Recall[class={label}]'] = metric_recall.compute(
            predictions=(predictions == label).astype(int),
            references=(labels == label).astype(int),
            zero_division=0
        )['recall']

    f1_weighted = metric_f1.compute(predictions=predictions, references=labels, average='weighted')
    f1_macro = metric_f1.compute(predictions=predictions, references=labels, average='macro')
    f1_micro = metric_f1.compute(predictions=predictions, references=labels, average='micro')

    precision_macro = metric_precision.compute(predictions=predictions, references=labels, average='macro', zero_division=0)
    recall_macro = metric_recall.compute(predictions=predictions, references=labels, average='macro', zero_division=0)

    return  {
        **accuracy,
        **precisions,
        **recalls,
        'f1_weighted': f1_weighted['f1'],
        'f1_micro': f1_micro['f1'],
        'f1_macro': f1_macro['f1'],
        'precision_macro': precision_macro['precision'],
        'recall_macro': recall_macro['recall'],
        'mae': mean_absolute_error,
        'mae_w': mean_absolute_error_weighted,
        'mean_cross_entropy_loss': mean_cross_entropy_loss,
        'perprexity': np.exp(mean_cross_entropy_loss),
        'mean_gaussian_cross_entropy_loss': mean_gaussian_cross_entropy_loss
        }


def construct_command_line_call(script: str, **kwargs) -> str:
    """
    Constructs a command line call for a Python script with the given arguments.

    This function takes a script name and a variable number of keyword arguments, and constructs a command line call that can be used to run the script with the given arguments.

    Args:
        script (str): The name of the Python script to run.
        **kwargs: Arbitrary keyword arguments to pass to the script.

    Returns:
        str: The constructed command line call.
    """
    command = f'python {script}'
    for key, value in kwargs.items():
        if isinstance(value, str):
            value = f"'{value}'"
        command += f' --{key} {value}'
    return command


def hash_dict(dictionary: dict) -> str:
    json_args = json.dumps(dictionary, sort_keys=True)
    hasher = blake2b(digest_size=10)
    hasher.update(json_args.encode())
    hash = hasher.hexdigest()
    return hash


def load_dataset(
        dataset_path: str,
        text_column: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        sampling_ratio: Optional[float] = None,
        overwrite: Optional[bool] = False,
        undersampling: Optional[bool] = False,
        u_seed: int= 40,
        max_l: int = 512
    ) -> DatasetDict:
    """
    Loads a dataset from disk, tokenizes it, and optionally returns a subset of it.

    This function loads a dataset from the given path, tokenizes it using the given tokenizer, and returns a subset of it based on the given sampling ratio.
    If a tokenized version of the dataset already exists on disk, it is loaded instead of tokenizing the dataset again. 
    If no sampling ratio is provided, the entire tokenized dataset is returned.

    Args:
        dataset_path (str): The path to the dataset.
        text_column (str): name of the column containing the text to tokenize
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer to use.
        sampling_ratio (float, optional): The ratio of the dataset to return. Must be between 0 and 1. If None, the entire dataset is returned.
        overwrite: forces to recreate the tokenized dataset, whether it existed or not

    Returns:
        DatasetDict: The tokenized and optionally sampled dataset.
    """
    vocab_hash = hash_dict(tokenizer.vocab)

    max_l = min(max_l, tokenizer.model_max_length)
    
    path_tokenized = os.path.join(dataset_path, f'vocab_{vocab_hash}_col_{text_column}_sr_{sampling_ratio}_us_{undersampling}_{u_seed}_maxL_{max_l}')
    assert text_column is not None, 'Missing the name of the column to tokenize'

    if overwrite:
        dataset = load_from_disk(dataset_path)
        dataset.cleanup_cache_files()
        tokenized_dataset = dataset.map(
            lambda x: tokenizer(x[text_column]), # we don't do any padding or truncation here, this is taken care of at a later stage
            batched=True,
            desc='Tokenizing dataset',
            load_from_cache_file = False
        )

     
        # removing too long sentences (if any) - since we append entity name and the end, we would not be able to do proper training/inference
        # NOTE: we could append names at the start of the sentence instead and just allow for losing informaiton at the end
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) <= max_l)
        tokenized_dataset.save_to_disk(path_tokenized)

    else:
        try:
            tokenized_dataset = load_from_disk(path_tokenized)
        except FileNotFoundError:
            dataset = load_from_disk(dataset_path)
            
            tokenized_dataset = dataset.map(
                lambda x: tokenizer(x[text_column]), # we don't do any padding or truncation here, this is taken care of at a later stage
                batched=True,
                desc='Tokenizing dataset'
            )
        
            # removing too long sentences (if any) - since we append entity name and the end, we would not be able to do proper training/inference
            # NOTE: we could append names at the start of the sentence instead and just allow for losing informaiton at the end
            tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) <= max_l)
    
            tokenized_dataset.save_to_disk(path_tokenized)

    if sampling_ratio is not None:
        tokenized_dataset = DatasetDict({
            k: v.select(range(int(sampling_ratio * v.num_rows)))
            for k, v in tokenized_dataset.shuffle(42).items()
        })

    #undersampling
    if undersampling:
        random.seed(u_seed)
        for k, v in tokenized_dataset.items():
            class_distribution = Counter(v['labels'])
            min_samples = min(class_distribution.values())
            undersampled_indices = []
            for label in class_distribution.keys():
                label_indices = [i for i, lbl in enumerate(v['labels']) if lbl == label]
                undersampled_indices.extend(random.sample(label_indices, min_samples))
            tokenized_dataset[k] = v.select(undersampled_indices)


    return tokenized_dataset


def perform_training(cli: bool, **parameters):
    if cli:
        command = construct_command_line_call(
            "tools/classification_training.py",
            **parameters
            )
        os.system(command)
    else:
        execute_training(**parameters)


def execute_training(
        model_checkpoint: str,
        dataset_path: str,
        text_column: str,
        output_dir: str,
        num_labels: int, # should match the unique categories in the labels column..
        gradual_unfreeze: bool = False,
        num_layers_to_train: Optional[int] = None,
        tokenizer_checkpoint: Optional[str] = None,
        perform_gaussian_smoothing: bool = True,
        use_class_weights: bool = True,
        early_stopping_patience: Optional[int] = 3,
        verbose_early_stopping: bool = True,
        dataset_sampling_ratio: float = 1,
        dataset_overwrite = False,
        discriminate: bool = True,
        dft_rate: float = 1.2, #only used if discriminate = True
        undersampling: bool = False,
        **trainer_kwargs):

    if gradual_unfreeze and ((early_stopping_patience or 0) > 0):
        print('gradual_unfreeze', gradual_unfreeze)
        print('early_stopping_patience', early_stopping_patience)
        raise ValueError('Gradual unfreezing cannot be used simultaneously with early stopping. Please disable one of these options.')


    if tokenizer_checkpoint is None:
        tokenizer_checkpoint = model_checkpoint
    transformers.set_seed(trainer_kwargs.get('seed', 40))

    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)


    num_encoder_layers = (
        getattr(model.config, 'n_layers', None)
        or getattr(model.config, 'num_hidden_layers', None)
        )

    assert num_encoder_layers is not None, 'Failed to determine the number of encoder layers'


    num_layers_to_train = num_layers_to_train or num_encoder_layers

    num_layers_to_train = min(
        num_layers_to_train or (num_encoder_layers + 1),  # If not defined train all encoder layers plus one for classifier
        (num_encoder_layers + 1))  # Do not attempt to train more layers than the model has


    if num_layers_to_train is None:
        raise ValueError('Could not determine the number of encoder layers.')

    dataset = load_dataset(
        dataset_path,
        text_column = text_column,
        tokenizer = tokenizer,
        sampling_ratio = dataset_sampling_ratio,
        overwrite = dataset_overwrite,
        undersampling = undersampling,
        u_seed = trainer_kwargs.get('seed', 40),
        max_l = model.config.max_position_embeddings
        )

    
    # by default, the 'labels' column is always used for training
    assert 'labels' not in dataset['train'], "Column 'labels' must be in the input dataset for traininig."

    
    for key, value in DEFAULT_TRAINING_ARGS.items():
        trainer_kwargs.setdefault(key, value)


    if trainer_kwargs.get('logging_steps', None) is None:
        trainer_kwargs['logging_steps'] = \
            len(dataset["train"]) // trainer_kwargs['per_device_train_batch_size']

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, return_tensors='pt') # padding=True padding to max in the batch

    training_args = TrainingArguments(
        output_dir=output_dir,
        **trainer_kwargs
    )

    
    optimizer = create_raven_optimizer(training_args, model, discriminate=discriminate,dft_rate = dft_rate)
    lr_sheduler = None
    optimizers = optimizer, lr_sheduler
    
    callback_functions = []
    if (early_stopping_patience is not None) and (early_stopping_patience > 0):
        early_stopping_callback: TrainerCallback = RavenEarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            verbose=verbose_early_stopping)
        callback_functions.append(early_stopping_callback)

    if gradual_unfreeze:
        freezer_callback = RavenFreezingCallback(num_layers_to_train, debug=True)
        callback_functions.append(freezer_callback)


    trainer = RavenTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_classification_metrics,
        callbacks=callback_functions,
        perform_gaussian_smoothing=perform_gaussian_smoothing,
        use_class_weights=use_class_weights,
        optimizers=optimizers,
    )
   
    trainer.train()


def get_decay_parameters(module: nn.Module):
    all_parameters = transformers.trainer_pt_utils.get_parameter_names(
        module, forbidden_layer_types=[nn.LayerNorm])
    decay_parameters = [name for name in all_parameters if "bias" not in name]
    return decay_parameters


def create_raven_optimizer(
        args: TrainingArguments,
        model: PreTrainedModel,
        discriminate: bool = False,
        dft_rate: float = 1.2):

    if discriminate:
        lr = args.learning_rate
        num_layers = model.config.num_hidden_layers
        decay_parameter_names = {}
        optimizer_grouped_parameters = []

        model_module_names = [name for name, _ in model.named_children()]

        if 'dropout' in model_module_names:
            model_module_names.remove('dropout')

        main_module_name = model_module_names.pop(0)

        main_module = getattr(model, main_module_name)
        main_module_module_names = [name for name, _ in main_module.named_children()]

        assert 'embeddings' == main_module_module_names.pop(0), 'The lowest module is not Embeddings'

        decay_parameter_names[f'{main_module_name}.embeddings'] = get_decay_parameters(main_module.embeddings)
        embeddings_decay = {
            'params': [
                p for n, p in list(main_module.embeddings.named_parameters())
                if (n in decay_parameter_names[f'{main_module_name}.embeddings'] and p.requires_grad)],
            'weight_decay': args.weight_decay,
            'lr': lr / (dft_rate ** (num_layers + 1))
            }
        embeddings_nodecay = {
            'params': [
                p for n, p in list(main_module.embeddings.named_parameters())
                if (n not in decay_parameter_names[f'{main_module_name}.embeddings'] and p.requires_grad)],
            'weight_decay': 0.0,
            'lr': lr / (dft_rate ** (num_layers + 1))
            }
        optimizer_grouped_parameters.append(embeddings_decay)
        optimizer_grouped_parameters.append(embeddings_nodecay)


        encoder_layer_name = main_module_module_names.pop(0)
        assert encoder_layer_name in ['transformer', 'encoder'], 'Encoder not the second module!'
        decay_parameter_names[f'{main_module_name}.{encoder_layer_name}'] = {}

        for i in range(num_layers):
            module = getattr(main_module, encoder_layer_name).layer[i]
            decay_parameter_names[f'{main_module_name}.{encoder_layer_name}'][i] = get_decay_parameters(module)
            encoder_decay = {
                'params': [p for n, p in list(module.named_parameters()) if (n in decay_parameter_names[f'{main_module_name}.{encoder_layer_name}'][i] and p.requires_grad)],
                'weight_decay': args.weight_decay,
                'lr': lr / (dft_rate ** (num_layers - i))}
            encoder_nodecay = {
                'params': [p for n, p in list(module.named_parameters()) if (n not in decay_parameter_names[f'{main_module_name}.{encoder_layer_name}'][i] and p.requires_grad)],
                'weight_decay': 0.0,
                'lr': lr / (dft_rate ** (num_layers - i))}
            optimizer_grouped_parameters.append(encoder_decay)
            optimizer_grouped_parameters.append(encoder_nodecay)

        if len(main_module_module_names) != 0:
            assert 'pooler' == main_module_module_names.pop(0), f'Pooler not in main model {main_model}'
            decay_parameter_names[f'{main_module_name}.pooler'] = get_decay_parameters(main_module.pooler)
            pooler_decay = {
                'params': [
                    p for n, p in list(main_module.pooler.named_parameters())
                    if (n in decay_parameter_names[f'{main_module_name}.pooler'] and p.requires_grad)],
                'weight_decay': args.weight_decay,
                'lr': lr
                }
            pooler_nodecay = {
                    'params': [
                        p for n, p in list(main_module.pooler.named_parameters())
                        if (n not in decay_parameter_names[f'{main_module_name}.pooler'] and p.requires_grad)],
                    'weight_decay': 0.0,
                    'lr': lr
                    }
            optimizer_grouped_parameters.append(pooler_decay)
            optimizer_grouped_parameters.append(pooler_nodecay)

        for post_encoder_module_name in model_module_names:
            post_encoder_module = getattr(model, post_encoder_module_name)
            decay_parameter_names[post_encoder_module_name] = get_decay_parameters(post_encoder_module)

            classifier_decay = {
                'params': [
                    p for n, p in list(post_encoder_module.named_parameters())
                    if (n in decay_parameter_names[post_encoder_module_name] and p.requires_grad)],
                'weight_decay': args.weight_decay,
                'lr': lr
                }
            classifier_nodecay = {
                'params': [
                    p for n, p in list(post_encoder_module.named_parameters())
                    if (n not in decay_parameter_names[post_encoder_module_name] and p.requires_grad)],
                'weight_decay': 0.0,
                'lr': lr
                }
            optimizer_grouped_parameters.append(classifier_decay)
            optimizer_grouped_parameters.append(classifier_nodecay)

    else:
        decay_parameter_names = get_decay_parameters(model)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if (n in decay_parameter_names and p.requires_grad)],
                "weight_decay": args.weight_decay,},
            {
                "params": [p for n, p in model.named_parameters() if (n not in decay_parameter_names and p.requires_grad)],
                "weight_decay": 0.0,
            },
        ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    all_learnable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimized_parameters = [p for pg in optimizer.param_groups for p in pg['params']]
    assert len(all_learnable_parameters) == len(optimized_parameters), 'Not all parameters prepared for optimization'
    return optimizer 



class RavenTrainer(Trainer):

    def __init__(self, *args,
                 use_class_weights: bool = False,
                 perform_gaussian_smoothing: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.perform_gaussian_smoothing = perform_gaussian_smoothing
        
        if use_class_weights:
            self.class_weights = self.estimate_class_weights()  # .to(self.model.device)
        else:
            self.class_weights = None


    def estimate_class_weights(self):
        if self.train_dataset is None:
            return None
        
        label_counts = Counter(self.train_dataset["labels"])
        class_weights = [self.train_dataset.num_rows / (label_counts[label] or 1) for label in range(self.model.num_labels)]
        return torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(
            self,
            model: AutoModelForSequenceClassification,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False
            ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        if self.class_weights is None:
            loss_fct = nn.CrossEntropyLoss()
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))


        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.perform_gaussian_smoothing:
            num_classes = logits.size(-1)
            labels_gaussian = gaussian_labels(
                labels, num_classes)
            loss = loss_fct(logits, labels_gaussian)
        else:

            loss = loss_fct(logits, labels)

        if return_outputs:
            return (loss, outputs)
        return loss

   

def main():
    parser = argparse.ArgumentParser(description="Training script for your model")

    # Define command-line arguments
    parser.add_argument("--model_checkpoint", required=True, type=str, help="Path or name of the model checkpoint")
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to the training dataset")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to the training dataset")
    parser.add_argument("--num_labels", required=True, type=int, help="Number of sentiment labels")

    # Parse command-line arguments
    required_args, other_kwargs_list = parser.parse_known_args()  # parse_args()
    required_args = vars(required_args)

    other_kwargs_dict = {}
    for i in range(0, len(other_kwargs_list), 2):
        key = other_kwargs_list[i][2:]
        value = other_kwargs_list[i + 1]
        # Try to convert value to int or float
        try:
            value = eval(value)
        except NameError:
            # If conversion fails, leave as string
            pass
        other_kwargs_dict[key] = value
    execute_training(
        **required_args,
        **other_kwargs_dict
    )

if __name__ == "__main__":
    main()
