import asyncio
import functools
import inspect
import json
import os
from datetime import datetime
from hashlib import blake2b
from typing import Any, Callable, Optional

import pandas as pd


def result_attribute_to_experiment_hashes(
        path_result_hash: str,
        **attributes: dict[str, Any]
        ) -> list[str]:
    """Find the experiment hashes based on the contents of result files.

    Args:
        path_result_hash (str): Path to the location where result files are hashed
        **attributes

    Returns:
        list[str]: Returns a list of strings representing the hashes of
        experiments where the results contain attributes with specified values
    """
    result_files = [f for f in os.listdir(path_result_hash) if f.endswith('.json')]
    experiment_hashes = []
    for result_file in result_files:
        with open(path_result_hash + result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            all_attributes_match = True
            for attribute, value in attributes.items():
                if result_data[attribute] != value:
                    all_attributes_match = False
                    break
            if all_attributes_match:
                experiment_hashes.append(result_file.split('.')[0])
    return experiment_hashes


def experiment_hash_to_experiment_calls(
        experiment_hash: str,
        path_call_hash: str) -> list[dict[str, Any]]:
    """Load the function call data for a specified experiment hash.
    It returns a list in the unlikely case there are multiple calls with the
    same hash. The only case this could happen if there are two folders for two
    different functions and the parameters sent to those functions were identical.

    Args:
        experiment_hash (str): Hash of the experiment
        path_call_hash (str): Location of function call hashes.

    Returns:
        list[dict[str, Any]]: List of call parameters.
    """
    experiment_calls = []
    for root, _, call_files in os.walk(path_call_hash):
        for call_file in call_files:
            if not call_file.endswith('.json'):
                continue
            call_file_path = os.path.join(root, call_file)
            with open(call_file_path, 'r', encoding='utf-8') as f:
                call_data = json.load(f)
            if call_data['experiment_hash'] == experiment_hash:
                experiment_calls.append(call_data)
    return experiment_calls


def experiment_hash_to_experiment_results(
        experiment_hash: str,
        path_result_hash: str) -> list[dict[str, Any]]:
    """Load experiment results for a given experiment hash.

    Args:
        experiment_hash (str): Experiment hash.
        path_result_hash (str): Location where experiment results are stored.

    Returns:
        list[dict[str, Any]]: A list of experiment results.
    """
    experiment_results = []
    for root, _, result_files in os.walk(path_result_hash):
        for result_file in result_files:
            if result_file == f'{experiment_hash}.json':
                result_file_path = os.path.join(root, result_file)
                with open(result_file_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                experiment_results.append(result_data)
    return experiment_results


def serialize_arg(arg: Any) -> Any:
    """
    Serialize an argument for JSON encoding. If the argument is a callable (function),
    its name is used instead of the memory location.

    Args:
        arg (any): The argument to serialize.

    Returns:
        any: The serialized argument.
    """
    if callable(arg):
        return f"callable:{arg.__module__}.{arg.__qualname__}"
    elif isinstance(arg, pd.DataFrame):
        return f"dataframe:{hash(pd.util.hash_pandas_object(arg).sum())}"
    return arg


#TODO: Separate default variables, and look for changes in default variables.
#TODO: Do checksums for functions and store function source code in a file.
#TODO: Change how kwargs are stored in the experiment call hash, now its
# "kwargs": {"key": value, ...}, it should be "key": value, ...
def experiment_tracker(
        path_call_hash: str,
        path_result_hash: Optional[str] = None,
        return_hash: bool = False):
    """
    A decorator factory for tracking experiments. Records the function call with its arguments
    and the result to a JSON file. Supports both synchronous and asynchronous functions.

    Args:
        path_call_hash (str): Base directory path where function call hashes will be saved.
        path_result_hash (Optional[str], optional): Base directory path where function result hashes will be saved. Defaults to None.
        return_hash (bool, optional): If True, appends the experiment hash to the arguments. Defaults to False.

    Returns:
        Callable: A decorator that can be applied to functions to track their execution.
    """
    def decorator_track_experiment(func: Callable) -> Callable:
        """
        The actual decorator that wraps the function to track its execution.

        Args:
            func (Callable): The function to be tracked by the decorator.

        Returns:
            Callable: The wrapped function.
        """
        # Wrapper for asynchronously executed functions
        async def async_retrieve_results(*args, **kwargs):
            return await func(*args, **kwargs)

        # Wrapper for synchronously executed functions
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """
            A synchronous wrapper function that tracks the execution of the wrapped function.
            
            The `sync_wrapper` deals with synchronous function calls, generating a unique hash
            for each call based on its arguments, and recording the call and its results to a file.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                The result of the wrapped function call.
            """
            # Retrieve the function's signature and bind the passed arguments to it
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Convert the arguments to a dictionary
            args_dict = bound_args.arguments

            # Serialize the dictionary to a JSON formatted string
            # Note: This assumes that all arguments are JSON serializable
            json_args = json.dumps(args_dict, sort_keys=True, default=serialize_arg)

            # Hash the JSON string to create a unique identifier

            hasher = blake2b(digest_size=10)
            hasher.update(json_args.encode())
            experiment_hash = hasher.hexdigest()

            if return_hash:
                kwargs['experiment_hash'] = experiment_hash
                args_dict['experiment_hash'] = experiment_hash
                json_args = json.dumps(args_dict, sort_keys=True, default=serialize_arg)

            # Check if the function call is already recorded
            full_path = path_call_hash + f"{func.__name__}/{experiment_hash}.json"
            if not os.path.isfile(full_path):
                # Save the experiment parameters and hash to a JSON file
                experiment_data = {
                    'parameters': json.loads(json_args),
                    'func_name': func.__name__,
                    'date': datetime.now().isoformat(),
                    'experiment_hash': experiment_hash
                }
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(experiment_data, f, indent=4)

            if path_result_hash:
                result_path = f"{path_result_hash}{experiment_hash}.json"
                try:
                    results = load_results_from_file(result_path)
                except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
                    results = func(*args, **kwargs)
                    save_results_to_file(result_path, results)
            else:
                results = func(*args, **kwargs)
            return results

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """
            An asynchronous wrapper function that tracks the execution of the wrapped async function.

            The `async_wrapper` deals with asynchronous function calls, similar to `sync_wrapper`,
            but is tailored to support asynchronous execution flow.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                The result of the wrapped async function call.
            """
            # Retrieve the function's signature and bind the passed arguments to it
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Convert the arguments to a dictionary
            args_dict = bound_args.arguments

            # Serialize the dictionary to a JSON formatted string
            # Note: This assumes that all arguments are JSON serializable
            json_args = json.dumps(args_dict, sort_keys=True, default=serialize_arg)

            # Hash the JSON string to create a unique identifier

            hasher = blake2b(digest_size=10)
            hasher.update(json_args.encode())
            experiment_hash = hasher.hexdigest()

            if return_hash:
                kwargs['experiment_hash'] = experiment_hash
                args_dict['experiment_hash'] = experiment_hash
                json_args = json.dumps(args_dict, sort_keys=True, default=serialize_arg)

            # Check if the function call is already recorded
            full_path = path_call_hash + f"{func.__name__}/{experiment_hash}.json"
            if not os.path.isfile(full_path):
                # Save the experiment parameters and hash to a JSON file
                experiment_data = {
                    'parameters': json.loads(json_args),
                    'func_name': func.__name__,
                    'date': datetime.now().isoformat(),
                    'experiment_hash': experiment_hash
                }
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(experiment_data, f, indent=4)

            if path_result_hash:
                result_path = f"{path_result_hash}{experiment_hash}.json"
                try:
                    results = load_results_from_file(result_path)
                except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
                    results = await async_retrieve_results(*args, **kwargs)
                    save_results_to_file(result_path, results)
            else:
                results = await async_retrieve_results(*args, **kwargs)
            return results

        # Determine if the original function is async and return the appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator_track_experiment

def load_results_from_file(result_path):
    """
    Load results from a JSON file.

    This function reads a file from the given path and loads its content as a JSON object. It can be modified to load
    data saved in numpy's binary format by uncommenting the alternative block and commenting out the JSON loading line.

    Parameters:
    - result_path (str): The path to the file containing the results.

    Returns:
    - object: The content of the file parsed from JSON format. Can be a dictionary, list, etc., depending on the file content.
    """
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    # Alternatively, if using np.save/np.load:
    # return np.load(result_path, allow_pickle=True).item()


def save_results_to_file(result_path, results):
    """
    Save results to a JSON file.

    This function saves the provided object into a file at the specified path in JSON format. It ensures that the
    directory for the file exists. If not, it creates the directory structure before writing the file. Similar to loading,
    this function can be adapted to save data in numpy's binary format by using the commented numpy block.

    Parameters:
    - result_path (str): The path where results should be saved.
    - results (object): The data object to save, which can be a dictionary, list, etc.

    """
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    # Alternatively, if using np.save/np.load:
    # np.save(result_path, results)
