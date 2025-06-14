# Adaptive Timestep Sampling

This project explores adaptive timestep sampling for image generation with Denoising Diffusion Probabilistic Models (DDPMs) from a frequency-domain perspective. The initial goal was to investigate whether sampling strategies could be improved by incorporating image spectral characteristics. It also examines resolution compensation through spectral analysis, compares the results with existing research, and discusses potential reasons for observed differences.

⚠️ Note: This is an exploratory project intended to develop intuitions and test ideas. No formal model training or benchmarking was performed.

## Prerequisites

- Python 3.13
- Hugging Face account
- Access to Flux.1-dev model

## Installation

### Python Dependencies

You have three options to install the required dependencies:

1. Using pip:
   ```bash
   pip install -r requirements.txt --index-strategy unsafe-best-match
   ```

2. Using pipenv:
   ```bash
   pip install pipenv
   pipenv install
   ```
   This will create a virtual environment in your home directory with all required dependencies from the Pipfile.

3. Using uv:
   ```bash
   pip install uv
   uv sync
   ```
   To install **uv** without using `pip`, see the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
   
   This will create a virtual environment in the project directory with all required dependencies from `uv.lock`.

### Hugging Face Setup

The VAE weights are not included in this repository and will be downloaded during execution. You'll need to:

1. Create a [Hugging Face account](https://huggingface.co/)
2. Request access to the [Flux.1-dev model](https://huggingface.co/black-forest-labs/FLUX.1-dev)
3. [Create an access token](https://huggingface.co/docs/hub/security-tokens)

## Running the Notebook

1. Select the correct Python interpreter in the notebook.
2. During first execution, provide your Hugging Face access token in the login widget
   - The token will be cached at `~/.cache/huggingface/token`
   - Subsequent runs won't require token input

For more authentication details, see the [Hugging Face documentation](https://huggingface.co/docs/huggingface_hub/package_reference/authentication#huggingface_hub.login).