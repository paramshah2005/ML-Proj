# Train Nystroem + SGD on Google Colab

The Nystroem model is optional in the Streamlit dashboard. The existing deployed app continues to use the three completed local models until all four Nystroem artifact files are present.

## 1. Push the training code

Commit and push the project through VS Code before opening Colab. The raw dataset remains excluded by `.gitignore` and must be uploaded separately.

## 2. Clone the project in Colab

Create a new Google Colab notebook and run:

```python
!git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
%cd YOUR-REPOSITORY
!python -m pip install -r requirements.txt
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with the GitHub repository details.

## 3. Upload the dataset

```python
from google.colab import files

files.upload()
```

Select `datasetIRB2400.csv`. It must appear in the repository root beside `streamlit_app.py`.

## 4. Optional short trial

This verifies the Colab environment without running the full experiment:

```python
!python src/train_nystroem_sgd.py --max-fit-samples 10000 --n-components 100
```

The trial artifacts are replaced by the full run in the next step.

## 5. Train on every fitting row

```python
!python src/train_nystroem_sgd.py
```

The default full run uses:

- 255,000 train-plus-validation rows
- 500 Nystroem components
- RBF kernel approximation with deterministic seed 42
- six independent epsilon-insensitive SGD regressors
- the same untouched 45,000-row test set as the other models

Progress appears directly below the Colab cell and is also written to `artifacts/nystroem_sgd_training.log`.

## 6. Download the completed artifacts

```python
!zip -r nystroem_result.zip artifacts/nystroem_sgd artifacts/model_comparison.csv
```

```python
from google.colab import files

files.download("nystroem_result.zip")
```

## 7. Add the result to the local project

Extract the ZIP so these paths exist locally:

```text
D:\ML Proj\artifacts\nystroem_sgd\model.joblib
D:\ML Proj\artifacts\nystroem_sgd\metrics.json
D:\ML Proj\artifacts\nystroem_sgd\per_joint_metrics.csv
D:\ML Proj\artifacts\nystroem_sgd\test_predictions.csv
D:\ML Proj\artifacts\model_comparison.csv
```

Commit and push those files. Streamlit will redeploy and automatically add **Nystroem + SGD** to the prediction, comparison, analysis, and method sections.

