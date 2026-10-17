# IRB2400 Joint Target Dashboard

This Streamlit dashboard predicts the six desired ABB IRB2400 joint angles from:

- desired position: `x`, `y`, `z`
- desired orientation: `yaw`, `pitch`, `roll`
- current joint state: `q1_in` through `q6_in`

It loads the existing linear regression, RBF-SVR, decision-tree, and optional Nystroem + SGD artifacts. It also compares their saved test metrics and visualizes held-out prediction errors.

After a prediction, an interactive 3D technical skeleton animates from the current joint state to the predicted target state. The view includes play and scrub controls, a desired tool-pose marker, and dataset-calibrated Cartesian and orientation error readouts. It is a kinematic visualization and does not perform collision or dynamics simulation.

## Run locally

Open PowerShell in `D:\ML Proj`, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

Streamlit should open `http://localhost:8501` automatically. If it does not, open that address in a browser.

Stop the server with `Ctrl+C` in the same PowerShell window.

## Files required by the hosted app

Keep these paths in the GitHub repository:

```text
streamlit_app.py
requirements.txt
.streamlit/config.toml
src/train_svr.py
src/train_nystroem_sgd.py
src/common.py
artifacts/model_comparison.csv
artifacts/linear_regression/*
artifacts/rbf_svr/*
artifacts/decision_tree/*
```

The dashboard automatically enables Nystroem + SGD when `artifacts/nystroem_sgd/` contains its model, metrics, per-joint metrics, and test predictions. See `COLAB_NYSTROEM.md` for the Google Colab workflow.

Training logs and the 300,000-row source dataset are excluded by `.gitignore`; the deployed dashboard uses the saved models and held-out prediction files instead.

## Host on Streamlit Community Cloud

1. Create a GitHub repository and push this project from `D:\ML Proj`.
2. Confirm that the `artifacts` subfolders above appear on GitHub. They are needed at runtime.
3. Sign in at `https://share.streamlit.io` with GitHub.
4. Select **Create app**.
5. Choose the repository and branch, then set the entrypoint file to `streamlit_app.py`.
6. In advanced settings, select Python 3.12.
7. Deploy and wait for dependency installation and startup to finish.

No secrets are required for this dashboard.

## First GitHub push

After creating an empty repository on GitHub, run these commands from `D:\ML Proj`. Replace the sample URL with your repository URL.

```powershell
git init
git add .
git commit -m "Add IRB2400 Streamlit dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

