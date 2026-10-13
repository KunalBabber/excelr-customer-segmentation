"""Small Streamlit entrypoint. Business logic lives in src/."""
from pathlib import Path
import pandas as pd
import streamlit as st
from src.pipeline import prepare
from src.models import train_all,evaluate
from src.views import overview,show_models,show_evaluation,show_segments

st.set_page_config(page_title="ExcelR Customer Segmentation",page_icon="📊",layout="wide")
st.title("ExcelR | Customer Segmentation")
st.caption("Five clustering models • Model evaluation • Customer profiles • PCA")
source=st.sidebar.file_uploader("Upload marketing_campaign.xlsx or CSV",type=["xlsx","csv"])
@st.cache_data(show_spinner=False)
def read_data(upload):
    return pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
@st.cache_data(show_spinner="Cleaning and preparing customers...")
def preprocess(df):return prepare(df)
@st.cache_data(show_spinner="Training and evaluating five clustering models...")
def fit(X):
    preds=train_all(X)
    return preds,evaluate(X,preds)
try:
    if source is not None: raw=read_data(source)
    elif Path("marketing_campaign.xlsx").exists(): raw=pd.read_excel("marketing_campaign.xlsx")
    else:
        st.info("Upload the marketing_campaign dataset in the sidebar to begin.");st.stop()
    profile,X=preprocess(raw)
    tab1,tab2,tab3,tab4=st.tabs(["Dataset & EDA","5 Model Building","Model Evaluation","Customer Segments & PCA"])
    with tab1:overview(profile)
    predictions,scores=fit(X)
    with tab2:show_models(predictions)
    with tab3:show_evaluation(scores)
    with tab4:show_segments(profile,X,predictions)
except Exception as exc:
    st.error(f"Unable to process dataset: {exc}")
    st.stop()
