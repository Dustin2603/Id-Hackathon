Optimize memory: add serving dataset prep, mmap model loading, downcast DF, SHAP opt-in

This PR implements the memory-saving changes needed to avoid Out Of Memory crashes
on Streamlit Community Cloud by reducing the in-process memory footprint of the app.

Changes:
- New: scripts/prepare_serving_dataset.py
  - CLI helper to produce dataset_serving.parquet (selects only app columns, downcasts dtypes, optional sampling)
- Updated: train_models.py
  - Added --produce-serving flag to produce dataset_serving.parquet after training
  - Save compressed joblib artifacts (compress=3)
  - Attempt to save native xgboost booster models (.xgb) for more efficient serving
- Updated: app.py
  - Prefer loading dataset_serving.parquet via @st.cache_data if present; fallback to existing parquets
  - Apply immediate downcast of dataframe dtypes after load to reduce memory
  - load_models() uses @st.cache_resource with joblib.load(..., mmap_mode='r'), explicit gc.collect(), and robust error handling
  - SHAP computations are gated behind an explicit checkbox and performed only for the single input row
  - Large plotting datasets are downsampled before building Plotly traces
- requirements-optional.txt (new)
  - Moves shap into an optional requirements file so deployments can skip installing it
- README additions with instructions to produce dataset_serving.parquet locally and guidance for deploying to Streamlit Cloud

Notes:
- The optimized dataset (dataset_serving.parquet) is NOT committed to the repository. The script creates it offline for deployment.
- The training script remains an offline script and should NOT be used in the Streamlit runtime.

How to generate the serving dataset (example):
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt
3) python scripts/prepare_serving_dataset.py --input dataset_enriched.parquet --output dataset_serving.parquet

Please review the scripts and tests (if any). If you want, I can open this PR on your behalf now.
