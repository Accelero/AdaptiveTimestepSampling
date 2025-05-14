# Adaptive Timestep Sampling

This project investigates optimized time step sampling techniques for training Denoising Diffusion Probabilistic Models (DDPMs). The core focus is improving training efficiency by adapting the sampling distribution of time steps using frequency domain analysis.

## ⚠️ Notebook Preview Notice

**GitHub preview does not render the notebook properly.**  
To view the notebook **download it and open it locally.**

## Prerequisites

- Python 3.13
- Hugging Face account
- Access to Flux.1-dev model

## Installation

### Python Dependencies

You have two options to install the required dependencies:

1. Using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. Using pipenv (recommended):
   ```bash
   pip install pipenv
   pipenv install
   ```
   This will create a virtual environment in your home directory with all required dependencies from the Pipfile.

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