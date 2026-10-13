# ExcelR Customer Segmentation — Modular Streamlit App

The original notebook is preserved locally but excluded from the public repository because its saved outputs contain customer-row previews. The web app has a **small `app.py`**; all preprocessing, model training, metrics and visualizations live in `src/`.

## Structure
- `app.py` — Streamlit entrypoint and navigation
- `src/pipeline.py` — notebook-aligned feature engineering, IQR capping, scaling
- `src/models.py` — five clustering models and three internal validation metrics
- `src/views.py` — Streamlit charts, tables, downloads
- `marketing_campaign.xlsx` — local example dataset, not committed
- `ExcelR_Customer_Segmentation_5_Models_Evaluation_REVISED.ipynb` — local analysis notebook, not committed because saved outputs include customer-row previews
- `requirements.txt` — runtime dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload the project files **preserving `src/` structure**.
2. Open https://share.streamlit.io and sign in.
3. Select the repository, branch, and main file path `app.py`.
4. Click Deploy. Keep the repository public only if you are authorized to publish the dataset.

## Important
The notebook hardcodes project year **2026** for Age and uses **3 clusters** for K-Means, Agglomerative, GMM and BIRCH. DBSCAN uses `eps=1.8` and `min_samples=10`. The app mirrors these choices. DBSCAN may yield one/no valid cluster; its internal metrics will then be undefined. The app is not a live hosted deployment until connected to your own GitHub/Streamlit account.
