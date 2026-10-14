# ExcelR Customer Segmentation — Notebook to Streamlit

1. Put your `marketing_campaign.xlsx` next to the notebook.
2. Open `ExcelR_Customer_Segmentation_Deployment.ipynb` in Jupyter and run **all cells in order**. The last cell generates `customer_segmentation.pkl`.
3. Copy the generated PKL into the same folder as `app.py`.
4. Install dependencies: `pip install -r requirements.txt`.
5. Run locally: `streamlit run app.py`.
6. For Streamlit Community Cloud: push `app.py`, `customer_segmentation.pkl`, `requirements.txt` to GitHub; open https://share.streamlit.io; choose the repo and `app.py`; deploy.

The app displays the saved notebook results, not new predictions. The notebook overwrites the 13 feature columns with standardized values, so the dashboard profiles are in standardized units. The notebook source analysis remains unchanged; only the deployment section was appended.

**Security:** Never load untrusted PKL files.
