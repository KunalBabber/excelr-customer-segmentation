# ExcelR Customer Segmentation Dashboard

Interactive Streamlit version of the ExcelR customer segmentation notebook. Supports five clustering algorithms: K-Means, Agglomerative, Gaussian Mixture Model, BIRCH, DBSCAN. Evaluates using Silhouette, Davies-Bouldin and Calinski-Harabasz indices. Includes PCA, segment profiling and CSV export.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free using Streamlit Community Cloud

1. Add this folder to the repository at `deploy/ExcelR_Streamlit_Customer_Segmentation/`.
2. Visit https://share.streamlit.io and sign in using GitHub.
3. Select the repository and branch, set the main file path to `deploy/ExcelR_Streamlit_Customer_Segmentation/app.py`, and deploy.
4. Upload an Excel/CSV dataset in the app. The example workbook is intentionally not included in the public repository.

## Data privacy

The bundled dataset is an example from the project. **Before making the GitHub repository public**, confirm you have permission to publish its customer data. You can omit the `.xlsx` file and use the upload control instead. Avoid uploading confidential customer records to a public hosted app.

## Notes

- Uses 2026 as the default age reference year, consistent with the original notebook; configurable in the sidebar.
- Outlier capping applies to selected modeling features, not identifiers. Original notebook capped every numeric column; this avoids altering IDs and years.
- Profiles show engineered values after optional outlier capping, not standardized units.
- DBSCAN can produce only one cluster or all noise; evaluation metrics then show N/A.
- The model with highest silhouette is highlighted, but this is not a guarantee of business suitability.
