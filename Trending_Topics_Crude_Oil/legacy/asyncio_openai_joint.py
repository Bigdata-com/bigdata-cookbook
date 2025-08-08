import asyncio
from httpx import ReadTimeout
from typing import Any, Generator, Iterable, Optional, Union

import openai
from tqdm.asyncio import tqdm as async_tqdm

from tools.experiment_tracker import experiment_tracker


def convert_to_dict(obj) -> Union[list[Any], dict[Any, Any], Any]:
    """
    A function converts an object to a serializable dictionary.
    This is primarily used to convert ChatCompletion object to a dictionary.

    Args:
        obj (Any): Object to convert to dictionary

    Returns:
        dict: a dictionary containing serializable elements.
    """
    if isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {key: convert_to_dict(value) for key, value in obj.__dict__.items()}
    else:
        return obj

async def run_prompts(
        prompts: list,
        parameters: dict,
        concurrency: int,
        api_key: str,
        path_call_hash: str,
        path_result_hash: Optional[str] = None,
        max_retries: int = 1,
        timeout: int = 60,
        retry_delay: int = 1,
        exponential_backoff: bool = False
        ):
    """
    Executes a series of prompts concurrently using an external API, with support for retries on failure.

    Args:
        prompts (list): A list of prompts to be sent to the API.
        parameters (dict): A dictionary of additional parameters to be sent with each API call.
        concurrency (int): The number of concurrent calls to the API allowed.
        api_key (str): The API key required for making the API calls.
        path_call_hash (str): The file path for storing a hash of each API call.
        path_result_hash (str, optional): The file path for storing the results of the API calls.
        max_retries (int, optional): The maximum number of retries on failure. Defaults to 1.
        timeout (int, optional): The timeout for the API calls in seconds. Defaults to 60.
        retry_delay (int, optional): The delay between retries in seconds. Defaults to 1.
        exponential_backoff (bool, optional): If True, the delay between retries will double with each attempt. Defaults to False.

    Returns:
        list: A list of results from the prompts processed.
    """

    # Define an asynchronous OpenAI client
    _client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout)  # max_retries=max_retries


    # Define the location where the function calls and ChatCompletion results will be stored
    hash_experiments_and_results = experiment_tracker(
        path_call_hash=path_call_hash,
        path_result_hash=path_result_hash)

    #@hash_results decorator will track the calls to query_openai function. It will
    # store the hashed calls with the set of arguments used. It will also store the
    # result of a given call as a JSON file and will load the JSON instead of a
    # repeated call
    # @retry(exceptions=ReadTimeout, tries=max_retries, delay=1)
    @hash_experiments_and_results
    async def query_openai(messages, **kwargs):
        """
        Queries an external API with a set of messages and any additional parameters.

        This function attempts the query, retries on specific exceptions, and applies exponential backoff if enabled.

        Args:
            messages (list): The messages to be sent as the API queries.
            kwargs (dict): Additional keyword arguments to be passed in the API call.

        Returns:
            dict: A dictionary representing the API response.
        """
        failed = True
        for n_attempt in range(1 + max_retries):
            try:
                completion = await _client.chat.completions.create(
                    messages=messages,
                    **kwargs)
                completion_dict = convert_to_dict(completion)
                return completion_dict
            except (ReadTimeout, openai.APITimeoutError):
                #print(f'ReadTimout attempt {n_attempt+1}/{1+max_retries}')
                if exponential_backoff:
                    await asyncio.sleep(retry_delay * 2**n_attempt)
                else:
                    await asyncio.sleep(retry_delay * n_attempt)
            except Exception as e:
                raise e

        if failed:
            exc = Exception('Failed to Query OpenAI.')
            raise exc


    async def identify_call(idx, messages, **parameters):
        """
        Wraps the query_openai function, mapping the results with their respective indexes.

        Args:
            idx (int): The index of the call.
            messages (list): The messages to be sent to the API.
            parameters (dict): Additional parameters for the API call.

        Returns:
            dict: A dictionary mapping the call index to its result.
        """
        return {idx: await query_openai(messages, **parameters)}


    async def process_generator(message_generator: list, parameters: dict, concurrent_tasks: int):
        """
        Asynchronously processes a list of messages through the query_openai function, managing concurrency.

        Args:
            message_generator (list): A list containing message sets for each API call.
            parameters (dict): Additional parameters to be passed to each API call.
            concurrent_tasks (int): The maximum number of concurrent API calls allowed.

        Returns:
            list: A list of results, ordered by the original indexes of the prompts.
        """
        tasks = set()
        results = dict()
        n_prompts = len(message_generator)
        n_sent = 0
        with async_tqdm(desc='Received Completions', total=n_prompts) as pbar_received:
           # Process each message set, creating an asynchronous task for each.
            for idx, messages in enumerate(message_generator):
                # raise Exception('About to run OpenAI!')
                task = asyncio.create_task(identify_call(idx, messages, **parameters))
                tasks.add(task)
                n_sent += 1
                # When the concurrency limit is reached, wait for at least one task to complete before proceeding.
                if len(tasks) >= concurrent_tasks:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    pbar_received.set_description(f'Sent: {n_sent}, Received', refresh=True)
                    pbar_received.update(len(done))
                    tasks.difference_update(done)
                    results.update({key: value for done_task in done for key, value in done_task.result().items()})

            remaining_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.update({key: value for reimaining_result in remaining_results for key, value in reimaining_result.items()})

            # Final progress update after all tasks are completed
            for _ in remaining_results:
                pbar_received.set_description(f'Sent: {n_sent}, Received', refresh=True)
                pbar_received.update(1)  # Update the progress bar for each remaining task
        return [results[k] for k in sorted(results)]

    # Check if there's an already running event loop, if so, wait for the process_generator coroutine, else run a new event loop.
    loop = asyncio.get_running_loop()
    if (loop is not None) and loop.is_running():
        print('Found a running asynchronous loop.')
        results = await process_generator(message_generator=prompts, parameters=parameters, concurrent_tasks=concurrency)

    else:
        results = asyncio.run(process_generator(message_generator=prompts, parameters=parameters, concurrent_tasks=concurrency))
    return results
