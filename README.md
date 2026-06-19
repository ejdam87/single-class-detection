# Car Detection

Car Detection ML pipeline

## Description

This project composes a deep learning pipeline for multi-car detection in a given picture. It emphasises the understanding of the underlying detection model - **FasterRCNN**.

## Data

Data used for training and evaluation are not public, but were infered from public [City Scapes](https://www.cityscapes-dataset.com/) dataset.

## Usage

In the `src` folder

### Training

```
uv run python training.py <dataset_path>
```

### Inference

```
uv run python inference.py <dataset_path> model.pt 
```

### Evaluation

```
uv run python evaluation.py DET <dataset_path_bboxes> .\output_predictions\
```
