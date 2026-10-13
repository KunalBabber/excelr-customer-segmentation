"""Five clustering models matching the notebook's Section 9 parameters."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,AgglomerativeClustering,Birch,DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
from sklearn.decomposition import PCA

def train_all(X):
    models={
      "K-Means":KMeans(n_clusters=3,random_state=42,n_init=20),
      "Agglomerative":AgglomerativeClustering(n_clusters=3,linkage="ward"),
      "Gaussian Mixture (GMM)":GaussianMixture(n_components=3,covariance_type="full",random_state=42,n_init=10),
      "BIRCH":Birch(threshold=0.8,n_clusters=3),
      "DBSCAN":DBSCAN(eps=1.8,min_samples=10),
    }
    return {name:model.fit_predict(X) for name,model in models.items()}

def evaluate(X, predictions):
    records=[]
    for name,labels in predictions.items():
        labels=np.asarray(labels)
        noise=int(np.sum(labels==-1))
        mask=labels!=-1 if name=="DBSCAN" else np.ones(len(labels),dtype=bool)
        x=np.asarray(X)[mask];y=labels[mask];k=len(np.unique(y))
        row={"Model":name,"Clusters":k,"Noise Points":noise,"Silhouette Score":np.nan,"Davies-Bouldin Index":np.nan,"Calinski-Harabasz Index":np.nan}
        if 2<=k<len(y):
            row.update({"Silhouette Score":silhouette_score(x,y),"Davies-Bouldin Index":davies_bouldin_score(x,y),"Calinski-Harabasz Index":calinski_harabasz_score(x,y)})
        records.append(row)
    return pd.DataFrame(records)

def project_pca(X):
    return PCA(n_components=2,random_state=42).fit_transform(X)
