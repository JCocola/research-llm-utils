# OpenAI Research Utilities

This repository contains a **clean and simple** set of utility functions and code snippets designed to work with the OpenAI Python API. Developed as part of my ongoing research and collaborative projects, these tools are used for storing and sharing work in progress.

> **Note:** This is a work in progress and not meant for production use. The code is provided as-is and may require additional testing and validation. For more advanced features, check the [safety-tooling GitHub repository](https://github.com/safety-research/safety-tooling).

## Features

- **Secrets Loader:** Load API keys from a plain text secrets file with minimal overhead.
- **Fine-Tuning Utilities:** Run fine-tuning jobs, log experiment details in both JSON and CSV formats, and capture training data statistics.
- **Inference Utilities:** Generate text, retrieve token probabilities, and process multiple prompts concurrently using the `SimpleRunner` class.

## Requirements

- Python 3.8+
- [openai-python](https://github.com/openai/openai-python)
- Additional dependencies: `numpy`, `pandas`, `tqdm`, etc.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
```

## Usage Examples

### 1. Secrets Loader

Load your API keys from a file and set them as environment variables:

```python
from secrets_loader import load_secrets
import os

# Load secrets from your secrets file
secrets = load_secrets("SECRETS")

# Set the OpenAI API key dynamically
os.environ["OPENAI_API_KEY"] = secrets.get("OPENAI_API_KEY1", "")
print("Loaded API key:", os.environ["OPENAI_API_KEY"])
```

### 2. Fine-Tuning Utilities

Run a fine-tuning job and log experiment details:

```python
from finetuning_utils import run_fine_tuning
from openai import OpenAI

# Create a preconfigured OpenAI client instance
client = OpenAI()

# Set training parameters
training_data_path = "training_data.jsonl"  # Your training data file (JSONL format)
model_id = "base-model-id"  # Your base model identifier
hyperparameters = {
    "n_epochs": 4,
    "batch_size": 8,
    "learning_rate_multiplier": 0.05,
}

# Run the fine-tuning job
job, fine_tuned_model = run_fine_tuning(client, training_data_path, model_id, hyperparameters)
print("Fine-tuning completed. New model ID:", fine_tuned_model)
```

### 3. Inference Utilities with SimpleRunner

Use `SimpleRunner` to generate text and obtain token probabilities:

#### a. Generate Text

```python
from inference_utils import SimpleRunner
from openai import OpenAI

# Initialize the client and SimpleRunner
client = OpenAI()
runner = SimpleRunner(client, model="gpt-3.5-turbo", top_k=5)

# Define a conversation for text generation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like today?"}
]

# Generate text from the model
generated_text = runner.get_text(messages)
print("Generated text:", generated_text)
```

#### b. Retrieve Token Probabilities

```python
# Get the top token probabilities for the next token
top_probs = runner.get_top_k_probs(messages)
print("Top token probabilities:", top_probs)
```

#### c. Process Multiple Prompts Concurrently

```python
def example_func(prompt: str) -> str:
    return runner.get_text([{"role": "user", "content": prompt}])

# List of prompts to process concurrently
kwargs_list = [
    {"prompt": "Tell me a joke."},
    {"prompt": "What is the capital of France?"}
]

# Process prompts concurrently using get_many
for kwargs, result in runner.get_many(example_func, kwargs_list):
    print("Input:", kwargs, "Output:", result)
```

## Research & Collaboration

This repository is a central store for my research code and a collaboration tool for my colleagues. The goal is to provide a simple, clean foundation that can be extended as needed. If you have suggestions, improvements, or want to contribute, please feel free to open an issue or submit a pull request.

## Additional Functionalities

For more advanced features and safety tooling, please check out the [safety-tooling GitHub repository](https://github.com/yourusername/safety-tooling).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.