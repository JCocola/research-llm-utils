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

## Research & Collaboration

This repository is a central store for my research code and a collaboration tool for my colleagues. The goal is to provide a simple, clean foundation that can be extended as needed. If you have suggestions, improvements, or want to contribute, please feel free to open an issue or submit a pull request.

## Additional Functionalities

For more advanced features and safety tooling, please check out the [safety-tooling GitHub repository](https://github.com/yourusername/safety-tooling).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.