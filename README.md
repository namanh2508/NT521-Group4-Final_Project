

# ExploitGen: Template-augmented Exploit Code Generation based on CodeBERT

This project implements the ExploitGen model described in the paper "ExploitGen: Template-augmented exploit code generation based on CodeBERT" published in The Journal of Systems & Software. The model generates exploit code from natural language descriptions using a template-augmented approach based on CodeBERT.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training Process](#training-process)
  - [Domain-Adaptive Pre-training (DAPT)](#domain-adaptive-pre-training-dapt)
  - [Task-Adaptive Pre-training (TAPT)](#task-adaptive-pre-training-tapt)
  - [ExploitGen Model Training](#exploitgen-model-training)
- [Evaluation](#evaluation)
- [Usage Examples](#usage-examples)
- [Citation](#citation)

## Project Overview

ExploitGen addresses the challenging task of automatically generating exploit code from natural language descriptions. The key innovations include:

- **Template Parser**: A rule-based component that extracts domain-specific tokens and replaces them with placeholders
- **Dual-Encoder Architecture**: Uses two encoders to process both raw and template-augmented natural language
- **Semantic Attention Layer**: Dynamically combines information from different layers of the encoders
- **Fusion Layer**: Effectively integrates template information with raw semantic information
- **Decoder**: Generates the final code sequence using Transformer's decoder architecture
- **Adaptive Pre-training**: Combines Domain-Adaptive Pre-training (DAPT) and Task-Adaptive Pre-training (TAPT) to create FG-CodeBERT
  
## Directory Structure

A detailed breakdown of the project's files and directories to help you navigate the codebase.

ExploitGen/

├── config.py # ⚙️ Central configuration file for paths, hyperparameters, and model settings.

├── model.py # 🧠 PyTorch model definitions (ExploitGen, SemanticAttention, FusionLayer, etc.).

├── utils.py # 🛠️ Utility functions for data loading, tokenization, and Template Parser.

├── dapt_training.py # 🚀 Domain-Adaptive Pre-training (DAPT) on the SPoC dataset.

├── tapt_training.py # 🚀 Task-Adaptive Pre-training (TAPT) on exploit code datasets.

├── train.py # 🏋️ Final training script using the two-stage strategy + FG-CodeBERT.

├── evaluate.py # 📊 Inference & evaluation on the test set.

├── gen_exploit_code_example.py # After running and training this model successfully, use this code to test

├── requirements.txt # 📦 Python dependencies.

├── README.md # 📖 Project documentation.


├── data/ # 📂 All datasets.

│ ├── spoc/

│ │ └── train/

│ │ └──── spoc-train.tsv # SPoC dataset for DAPT (manual download).

│ ├── python/

│ │ ├──── train.csv # Python exploit code — training set.

│ │ ├──── dev.csv # Python exploit code — dev/validation set.

│ │ └──── test.csv # Python exploit code — test set.

│ └── assembly/

│ ├──── train.csv # Assembly exploit code — training set.

│ ├──── dev.csv # Assembly exploit code — dev/validation set.

│ └──── test.csv # Assembly exploit code — test set.


├── codeBERT-dapt/ # 📂 Output from DAPT ( created by dapt_training.py).


├── fg_codebert_model/ # 📂 Final FG-CodeBERT from TAPT ( created by tapt_training.py).


├── checkpoint-epoch-{epoch_number}/ # 📂 Saved ExploitGen checkpoints from each epoch. ( created by train.py )

└── logs/ # 📂 Training logs. ( created by dapt_training.py and tapt_training.py )

├──── dapt/ # TensorBoard logs for DAPT.

└──── tapt/ # TensorBoard logs for TAPT.

***Note***: Directories like codeBERT-dapt/, fg_codebert_model/, checkpoint-epoch-{}/, and logs/ are automatically created during training.

## Installation

### Prerequisites
- Python 3.7 or higher
- PyTorch 1.8.0 or higher
- CUDA-compatible GPU (recommended for training)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/NTDXYG/ExploitGen.git
cd ExploitGen
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv\Scripts\activate  # On Windows
source venv/bin/activate      # On Linux
```

3. Install the required packages:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # Thay đổi phiên bản CUDA, Pytorch ở đây ( https://pytorch.org/ )
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Data Preparation

1. Create the data directory structure:
```bash
mkdir -p data/spoc
mkdir -p data/python
mkdir -p data/assembly
```

2. Download and prepare the SPoC dataset for DAPT:
   - Download the SPoC dataset from the official source ( https://github.com/sumith1896/spoc.git )
   - Extract and place `spoc-train.tsv` in the `data/spoc/` directory

3. Prepare the exploit code datasets:
   - Download the Python and Assembly exploit datasets from ( https://github.com/NTDXYG/ExploitGen ) 
   - Place the processed CSV files from above Python and Assembly folders in `data/python/` and `data/assembly/` directories

   The CSV files should have the following columns:
   - `raw_nl`: Original natural language description
   - `temp_nl`: Template-augmented natural language
   - `raw_code`: Original exploit code
   - `temp_code`: Template-augmented exploit code

## Training Process

The training process consists of three main stages:

### Domain-Adaptive Pre-training (DAPT)

This stage adapts the original CodeBERT model to the domain of competitive programming using the SPoC dataset.

```bash
python dapt_training.py
```

The pre-trained model will be saved to `./codeBERT-dapt/` directory.

### Task-Adaptive Pre-training (TAPT)

This stage further adapts the model to the specific task of exploit code generation.

```bash
python tapt_training.py
```

The fine-tuned FG-CodeBERT model will be saved to `./fg_codebert_model/` directory.

### ExploitGen Model Training

This stage trains the complete ExploitGen model using the two-stage training strategy.

```bash
python train.py
```

The trained model checkpoints will be saved to `./checkpoint-epoch-{epoch_number}/` directories.

## Evaluation

To evaluate the trained model:

```bash
python evaluate.py
```

This will:
1. Generate code for a sample natural language description
2. Evaluate the model on the test set (commented out by default)
3. Calculate BLEU-4, ROUGE-W, and Exact Match metrics

## Usage Examples
```bash
python gen_exploit_code_example.py
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yang2023exploitgen,
  title={ExploitGen: Template-augmented exploit code generation based on CodeBERT},
  author={Yang, Guang and Zhou, Yu and Chen, Xiang and Zhang, Xiangyu and Han, Tingting and Chen, Taolue},
  journal={The Journal of Systems \& Software},
  volume={197},
  pages={111577},
  year={2023},
  publisher={Elsevier}
}
```

## License

This project is licensed under UIT - NT521.Q12.ANTT - Group 08 

## Contact

[Email: 23520075@gm.uit.edu.vn](mailto:23520075@gm.uit.edu.vn)

[Email: 23520281@gm.uit.edu.vn](mailto:23520281@gm.uit.edu.vn)

[Email: 23521610@gm.uit.edu.vn](mailto:23521610@gm.uit.edu.vn)

[Email: 23521260@gm.uit.edu.vn](mailto:23521260@gm.uit.edu.vn)

## Acknowledgments

We thank the original authors of the paper and the contributors to the CodeBERT project. We also acknowledge the providers of the datasets used in this project.
