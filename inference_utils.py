import openai
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from openai import OpenAI


class SimpleRunner:
    def __init__(self, client: OpenAI, model: str, top_k: int = 5):
        """
        Initialize the runner with the OpenAI client, model name, and top_k parameter.

        Args:
            client (OpenAI): Preconfigured OpenAI client instance.
            model (str): Name of the OpenAI model to use.
            top_k (int): Number of top logprobs to request (default=5, max=20).
        """
        self.client = client
        self.model = model
        self.top_k = min(top_k, 20)  # OpenAI typically limits this to 20

    def logprob_probs(self, messages: List[Dict]) -> Dict[str, float]:
        """
        Retrieve raw probabilities for the next token using logprobs.

        Args:
            messages (List[Dict]): List of message dictionaries.

        Returns:
            Dict[str, float]: Mapping of token strings to their probability.
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=self.top_k,
        )

        try:
            # Retrieve top logprobs for the first token generated
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
            return {el.token: float(np.exp(el.logprob)) for el in logprobs}
        except (IndexError, AttributeError):
            print(f"Warning: Failed to get logprobs from {self.model}")
            return {}

    def get_text(
        self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 1
    ) -> str:
        """
        Generate text output from the model given a conversation history.

        Args:
            messages (List[Dict]): List of message dictionaries.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: The generated text.
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def get_probs(
        self,
        messages: List[Dict],
        outputs: List[str],
        postprocess: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate normalized probabilities for specified outputs.

        Args:
            messages (List[Dict]): List of message dictionaries.
            outputs (List[str]): List of outputs to retrieve probabilities for.
            postprocess (Optional[Callable[[str], str]]): Function to process outputs before comparison.

        Returns:
            Dict[str, float]: Mapping of each output to its normalized probability.
        """
        # Retrieve raw token probabilities
        probs_dict = self.logprob_probs(messages)

        # Apply postprocessing if provided
        if postprocess:
            clean_probs_dict = defaultdict(float)
            for key, val in probs_dict.items():
                clean_key = postprocess(key)
                clean_probs_dict[clean_key] += val
            probs_dict = dict(clean_probs_dict)

        # Extract probabilities for the specified outputs
        result = {output: probs_dict.get(output, 0) for output in outputs}

        # Normalize the probabilities
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    def get_top_k_probs(self, messages: List[Dict]) -> Dict[str, float]:
        """
        Retrieve the top K token probabilities without normalization.

        Args:
            messages (List[Dict]): List of message dictionaries.

        Returns:
            Dict[str, float]: Sorted dictionary of token probabilities (highest first).
        """
        return dict(
            sorted(
                self.logprob_probs(messages).items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def get_many(
        self, func: Callable, kwargs_list: List[Dict[str, Any]], max_workers: int = 10
    ) -> Generator[Tuple[Dict[str, Any], Any], None, None]:
        """
        Process multiple prompts concurrently.

        Args:
            func (Callable): Function to execute for each set of parameters.
            kwargs_list (List[Dict[str, Any]]): List of dictionaries containing arguments for the function.
            max_workers (int): Maximum number of concurrent worker threads.

        Yields:
            Generator[Tuple[Dict[str, Any], Any], None, None]: Yields tuples of the original kwargs and the function's result.
        """

        def get_data(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
            # Filter out tracking fields (keys starting with '_')
            func_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            try:
                result = func(**func_kwargs)
                return kwargs, result
            except Exception as e:
                print(f"Error processing kwargs {kwargs}: {str(e)}")
                return kwargs, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_data, kw) for kw in kwargs_list]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    kwargs_result, result = future.result()
                    if result is not None:
                        yield kwargs_result, result
                except Exception as e:
                    print(f"Error in future: {str(e)}")
