# Agentic Coding for Science Demo

This GitHub repository contains code for a demo on agentic coding for science. The associated slides can be found here:
[Agentic Coding for Science Slides](https://docs.google.com/presentation/d/1AvQg3AVCsG52QoDZwgQ3cTqrW1SsQVa8fjQ_EcEu5-M/edit?usp=sharing)

## Contents

- `demo_notebook.ipynb`: A Google Colab notebook demonstrating how to train a convolutional neural network (CNN) on the MNIST dataset using JAX and the Flax NNX API. It is optimized for Cloud TPUs. This is based off the [official Flax MNIST tutorial](https://flax.readthedocs.io/en/latest/mnist_tutorial.html).
- `main.py`: A standalone Python script that performs the same training, evaluation, inference, and model export as the notebook, suitable for running locally.
- `bonsai/`: A git submodule linking to the [JAX Bonsai repository](https://github.com/jax-ml/bonsai).

## Setup

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. **Fork and clone the repository**:
   ```bash
   git clone --recursive <your-fork-url>
   cd MNIST_JAX_demo
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```
   This will automatically install the correct Python version (3.12) and all dependencies.

3. **Run the training script**:
   ```bash
   uv run python main.py
   ```
   This trains a CNN on MNIST, saves training curves to `training_curves.png`, sample predictions to `predictions.png`, and exports the model to `saved_models/demo_model/`.

## Running on Google Colab (TPU)

1. Open `demo_notebook.ipynb` within Google Colab or Antigravity.
2. Ensure the Colab session is connected to a Google Colab TPU runtime.
3. Run the setup cells to install the necessary packages and restart the kernel when prompted.
4. The notebook will automatically mount your Google Drive to save data and output models.

## Using with Agentic Coding Tools

- Use Gemini Code Assist on the left panel to tag the notebook and ask questions about it.
- Inside the notebook, use `cmd + I` (for mac) or `ctrl + I` (for windows) in individual cells to add additional documentation where unclear.
- Use the Agent Manager (top right) to ask the agent to come up with a plan to create a new, separate version of `demo_notebook.ipynb` that uses an advanced architecture from bonsai, keeping in mind that the Google Colab TPU kernel can only see inside the Google Drive folder and that bonsai was git cloned in that folder earlier.
- Comment on the plan where necessary and allow the agent to execute once you are aligned.
- Continue to iterate as needed.
