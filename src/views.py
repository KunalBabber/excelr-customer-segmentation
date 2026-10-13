"""Streamlit display functions, separate from app entrypoint."""
import pandas as pd
import plotly.express as px
import streamlit as st
from .models import project_pca

def overview(df):
    a,b,c=st.columns(3);a.metric("Customers",len(df));b.metric("Columns",len(df.columns));c.metric("Missing values",int(df.isna().sum().sum()))
    st.dataframe(df.head(20),use_container_width=True)
    st.subheader("Spending distribution")
    st.plotly_chart(px.histogram(df,x="TotalSpend",nbins=35),use_container_width=True)

def show_models(predictions):
    for name,labels in predictions.items():
        with st.expander(name,expanded=name=="K-Means"):
            counts=pd.Series(labels).value_counts().sort_index().rename_axis("Cluster").reset_index(name="Customers")
            st.dataframe(counts,use_container_width=True)
            st.plotly_chart(px.bar(counts,x="Cluster",y="Customers",title=name+" — cluster sizes"),use_container_width=True)

def show_evaluation(scores):
    st.dataframe(scores.round(3),use_container_width=True)
    for metric in ["Silhouette Score","Davies-Bouldin Index","Calinski-Harabasz Index"]:
        st.plotly_chart(px.bar(scores,x="Model",y=metric,title=metric),use_container_width=True)
    st.caption("Higher Silhouette and Calinski-Harabasz are preferred; lower Davies-Bouldin is preferred. DBSCAN noise is excluded from metric calculations. Undefined metrics are blank.")

def show_segments(profile,X,predictions):
    choice=st.selectbox("Choose clustering model",list(predictions))
    labels=predictions[choice]
    result=profile.copy();result["Cluster"]=labels
    st.dataframe(result.groupby("Cluster")[["Income","Age","TotalSpend","TotalPurchases","Recency"]].mean().round(2),use_container_width=True)
    pca=project_pca(X)
    chart=pd.DataFrame({"PC1":pca[:,0],"PC2":pca[:,1],"Cluster":pd.Series(labels).astype(str)})
    st.plotly_chart(px.scatter(chart,x="PC1",y="PC2",color="Cluster",title=choice+" — PCA projection"),use_container_width=True)
    st.download_button("Download segmented customers (CSV)",result.to_csv(index=False).encode(),"segmented_customers.csv","text/csv")
