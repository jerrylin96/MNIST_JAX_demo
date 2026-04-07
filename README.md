# Agentic Coding for Science Demo

This repository teaches agentic coding workflows using a JAX/Flax image classification pipeline
on satellite imagery relevant to earth and atmospheric science.
The associated slides can be found here:
[Agentic Coding for Science Slides](https://docs.google.com/presentation/d/1AvQg3AVCsG52QoDZwgQ3cTqrW1SsQVa8fjQ_EcEu5-M/edit?usp=sharing)

## Contents

- `demo_notebook.ipynb`: A Google Colab notebook that trains a CNN on the
  [EuroSAT](https://github.com/phelber/EuroSAT) dataset — 27,000 Sentinel-2 satellite images
  covering 10 land-use classes across Europe. Built with JAX, Flax NNX, and optimized for Cloud TPUs.
- `bonsai/`: Git submodule linking to the [JAX Bonsai repository](https://github.com/jax-ml/bonsai),
  which contains advanced vision architectures (Vision Transformers, ConvNeXt, etc.) in JAX.
- `tasks/`: Tiered task descriptions for progressive skill development.

## Instructions

1. **Fork the repository**: Click the "Fork" button in the top right and navigate to your new fork.

2. **Clone your fork and setup submodules**:
   ```bash
   git clone --recursive <your-fork-url> <desired-path-here>
   ```

3. **Open Google Antigravity**: If you do not have the `agy` shortcut installed, choose
   *Open Folder* within Google Antigravity and navigate to the cloned path.
   ```bash
   agy <desired-path-here>
   ```

4. **Run `demo_notebook.ipynb` on TPU**:
   - Open `demo_notebook.ipynb` within Antigravity.
   - Connect to a Google Colab **TPU** runtime.
   - Run the setup cells to install packages (`jax[tpu]`, `flax`, `tensorflow`, `jaxlib`)
     and restart the kernel when prompted.
   - The notebook will mount your Google Drive for storage and model export.

5. **Understand the codebase with Gemini Code Assist**:
   - Tag the notebook and ask questions in the left panel.
   - Use `Cmd+I` / `Ctrl+I` in individual cells to add documentation where unclear.

6. **Choose a task from `tasks/` and work with the Agent**:
   - Pick a task file from the `tasks/` directory that matches your experience level.
   - Open the Agent Manager (top right) and paste the task description.
   - Review the plan the agent proposes, comment where needed, then let it execute.
   - Continue iterating — the agent can read its own prior work and self-correct.

## Task Levels

| File | Level | Focus |
|---|---|---|
| `tasks/task1_architecture_swap.md` | Beginner | Swap CNN for a bonsai architecture |
| `tasks/task2_scientific_eval.md` | Intermediate | Add domain-relevant evaluation and visualization |
| `tasks/task3_data_pipeline.md` | Intermediate | Improve the data pipeline for scientific correctness |
| `tasks/task4_uncertainty.md` | Advanced | Add uncertainty quantification for scientific use |
| `tasks/task5_multispectral.md` | Advanced | Extend to 13-channel multispectral Sentinel-2 input |
