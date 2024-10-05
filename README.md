# Bet On Me

This repository contains two machine learning pipelines focused on different types of data and models:
1. A pipeline for working with large language models (LLMs) and generating embeddings using transformer-based models.
2. A pipeline for binary classification tasks using the XGBoost algorithm for tabular datasets.

## Project Overview

### 1. LLM Embeddings Pipeline (`pipelineLLMEmbeddings.ipynb`)
This notebook focuses on utilizing transformer models to generate text embeddings, which can be used for various downstream tasks such as text classification, semantic search, or other NLP applications.

#### Features:
- Utilizes Hugging Face's `transformers` library to load pre-trained language models.
- Handles tensor operations and model inference using PyTorch.
- Flexible pipeline for generating text embeddings and performing NLP tasks.

#### Requirements:
- `torch`
- `transformers`
- `bitsandbytes`
- `accelerate`

#### Setup:
```bash
# Install the required dependencies
pip install torch transformers bitsandbytes accelerate
```

#### Usage:
Run the notebook step by step to load a model, tokenize input text, and generate embeddings. You can modify the input data or use pre-trained models for various NLP tasks.

### 2. XGBoost Binary Classification Pipeline (`pipelineXGBoost_0415.ipynb`)
This notebook is designed for tabular data and uses XGBoost for binary classification tasks. It includes data preprocessing, model training, evaluation, and visualization.

#### Features:
- Reads data from CSV files and preprocesses it.
- Uses `XGBoost` for binary classification tasks.
- Evaluates the model with metrics such as accuracy, F1 score, precision, recall, and a confusion matrix.
- Supports visualization of the decision tree and confusion matrix.

#### Requirements:
- `xgboost`
- `scikit-learn`
- `matplotlib`
- `seaborn`

#### Setup:
```bash
# Install the required dependencies
pip install xgboost scikit-learn matplotlib seaborn
```

#### Usage:
Run the notebook step by step to load your dataset, preprocess the data, train the XGBoost model, and evaluate the results. You can visualize the final decision tree and confusion matrix at the end of the process.

## Repository Structure

- `pipelineLLMEmbeddings.ipynb`: Pipeline for generating text embeddings using large language models.
- `pipelineXGBoost_0415.ipynb`: Binary classification pipeline using XGBoost on tabular data.

## How to Run the Notebooks
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies for each notebook.

3. Open and run the notebooks using Jupyter:
   ```bash
   jupyter notebook pipelineLLMEmbeddings.ipynb
   jupyter notebook pipelineXGBoost_0415.ipynb
   ```
