# OC-SOD: Revisiting Salient Object Detection from an Observer-Centric Perspective

This repository contains the official implementation of **OC-SODAgent** and the associated **OC-SOD Dataset** generation pipeline.

## 📋 Project Overview

The **OC-SODAgent** (Object-Centric Salient Object Detection Agent) is an iterative framework that:
1. Performs initial segmentation using **SAM2**.
2. Evaluates the quality and semantic alignment using a **Vision Language Model (VLM)**.
3. Refines the segmentation mask based on the VLM's feedback until the object is accurately isolated.

## 🏗️ Project Structure

The project is organized efficiently as a Python package `OC_SOD`:

- `src/core`: Core agent implementation (`OCSODAgent`).
- `src/models`: Model wrappers for SAM2 and VLM.
- `src/pipeline`: Dataset generation pipeline (Intent, Saliency, Preference).
- `src/prompts`: Large language model prompt templates.
- `src/utils`: Utility functions and visualization tools.
- `configs/`: Configuration files for models and generation.
- `examples/`: Example scripts for inference and visualization.

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/OC-SOD.git
   cd OC-SOD
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install SAM2**
   Please follow the official instructions from [SAM2 repository](https://github.com/facebookresearch/segment-anything-2) to install the model, or use:
   ```bash
   pip install git+https://github.com/facebookresearch/sam2.git
   ```

### Inference with OC-SODAgent

```python
from OC_SOD.src.core import OCSODAgent

# Initialize agent
# Note: Ensure you have configured your models in configs/
agent = OCSODAgent() 

# Run iterative refinement
masks, info = agent.iterative_refine(image, bboxes, intent="intent description")
```

### Dataset Generation

The `src/pipeline` module handles data generation.
See `src/pipeline/generation.py` for details.

```python
from OC_SOD.src.pipeline import DatasetGenerator

generator = DatasetGenerator(output_dir="data/output")
# generator.paco_lvis_intent_gen(...)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
