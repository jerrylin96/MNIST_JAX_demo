# Agentic Coding for Science Demo

This GitHub repository contains code for a demo on agentic coding for science. The associated slides can be found here:
[Agentic Coding for Science Slides](https://docs.google.com/presentation/d/1AvQg3AVCsG52QoDZwgQ3cTqrW1SsQVa8fjQ_EcEu5-M/edit?usp=sharing)

## Contents

- `demo_notebook.ipynb`: A Google Colab notebook demonstrating how to train a convolutional neural network (CNN) on the MNIST dataset using JAX and the Flax NNX API. It is optimized for Cloud TPUs. This is based off the [official Flax MNIST tutorial](https://flax.readthedocs.io/en/latest/mnist_tutorial.html).
- `bonsai/`: A git submodule linking to the [JAX Bonsai repository](https://github.com/jax-ml/bonsai).

## Setup Instructions

1. **Fork the repository**: Navigate to [Agentic Coding Demo](https://github.com/jerrylin/agentic_coding_demo) (or the correct GitHub URL) and click the "Fork" button in the top right.

2. **Clone your fork and setup submodules**: Clone your forked repository to your local machine:
   ```bash
   git clone --recursive <your-fork-url> <desired-path-here>
   ```

3. **Open Google Antigravity**: If you do not have the agy shortcut installed, simply choose open folder within Google Antigravity and navigate to the path you cloned to.
   ```bash
   agy <desired-path-here>
   ```

4. **Run the demo using Google Antigravity**:
   - Open `demo_notebook.ipynb` within Antigravity.
   - Ensure the Colab session is connected to a TPU runtime.
   - Run the setup cells to install the necessary packages (`jax[tpu]`, `flax`, `tensorflow`, `jaxlib`) and restart the kernel when prompted.
   - The notebook will automatically mount your Google Drive to save data and output models.
