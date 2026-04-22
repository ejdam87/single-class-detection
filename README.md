# single-class-detection
Single Class Detection ML pipeline

```bash
uv pip install --python /home/jovyan/prostate-cancer/.venv/bin/python ipykernel
/home/jovyan/prostate-cancer/.venv/bin/python -m ipykernel install --user --name prostate --display-name "Python (prostate)"
```

training:

```
uv run python training.py ..\data_det_public\data_det_public\ 
```

inference:

```
uv run python inference.py ..\data_det_public\data_det_public\ model.pt 
```

eval:

```
uv run python evaluation.py DET ..\data_det_public\data_det_public\bbox\ .\output_predictions\
```
