"""ExcelR Customer Segmentation dashboard, adapted from the user's notebook."""
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

st.set_page_config(page_title='ExcelR | Customer Segmentation', page_icon='📊', layout='wide')
st.title('Customer Segmentation | ExcelR')
st.caption('Five unsupervised clustering models · evaluation · segment profiling · PCA')
FEATURES = ['Income','Age','Children','Recency','TotalSpend','TotalPurchases','NumDealsPurchases','NumWebVisitsMonth','TotalCampaignAccepted','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','Complain']
SPEND = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
PURCHASES = ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
CAMPAIGNS = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']
REQUIRED = ['Year_Birth','Income','Kidhome','Teenhome','Recency','NumDealsPurchases','NumWebVisitsMonth','Complain'] + SPEND + PURCHASES + CAMPAIGNS

@st.cache_data(show_spinner=False)
def load_data(file_bytes, filename):
    if filename.lower().endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))

def preprocess(raw, year, cap_outliers):
    df = raw.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    for c in set(REQUIRED):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Age'] = year - df['Year_Birth']
    df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
    df['Income'] = df['Income'].fillna(df['Income'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['TotalSpend'] = df[SPEND].sum(axis=1)
    df['TotalPurchases'] = df[PURCHASES].sum(axis=1)
    df['Children'] = df[['Kidhome','Teenhome']].sum(axis=1)
    df['TotalCampaignAccepted'] = df[CAMPAIGNS].sum(axis=1)
    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())
        if df[col].isna().any():
            raise ValueError(f'Column {col} has no usable numerical values.')
    # Match notebook's IQR capping for modeling variables without corrupting identifiers.
    if cap_outliers:
        for col in FEATURES:
            q1, q3 = df[col].quantile([0.25,0.75])
            iqr = q3-q1
            df[col] = df[col].clip(q1-1.5*iqr,q3+1.5*iqr)
    X = StandardScaler().fit_transform(df[FEATURES])
    return df, X

def evaluate(name, X, labels, exclude_noise=False):
    labels = np.asarray(labels)
    noise = int((labels == -1).sum())
    mask = labels != -1 if exclude_noise else np.ones(len(labels), dtype=bool)
    clean_labels, clean_X = labels[mask], X[mask]
    clusters = len(np.unique(clean_labels))
    result = {'Model':name,'Clusters':clusters,'Noise Points':noise,'Silhouette Score':np.nan,'Davies-Bouldin Index':np.nan,'Calinski-Harabasz Index':np.nan}
    if 2 <= clusters < len(clean_labels):
        result.update({'Silhouette Score':silhouette_score(clean_X,clean_labels),'Davies-Bouldin Index':davies_bouldin_score(clean_X,clean_labels),'Calinski-Harabasz Index':calinski_harabasz_score(clean_X,clean_labels)})
    return result

st.sidebar.header('Dataset & settings')
upload = st.sidebar.file_uploader('Upload marketing campaign dataset',type=['xlsx','csv'])
use_sample = st.sidebar.checkbox('Use bundled ExcelR dataset',value=True)
year = st.sidebar.number_input('Reference year for customer age',min_value=2015,max_value=2100,value=2026)
k = st.sidebar.slider('Clusters for K-Means / GMM / Ward / BIRCH',2,8,3)
eps = st.sidebar.slider('DBSCAN eps',0.2,5.0,1.8,0.1)
min_samples = st.sidebar.slider('DBSCAN min_samples',2,40,10)
cap_outliers = st.sidebar.checkbox('Apply IQR outlier capping',value=True)
if upload is not None:
    data_bytes, filename = upload.getvalue(), upload.name
elif use_sample:
    try:
        with open('marketing_campaign.xlsx','rb') as f: data_bytes = f.read()
        filename = 'marketing_campaign.xlsx'
    except FileNotFoundError:
        st.warning('Bundled dataset missing. Upload an Excel/CSV file.');st.stop()
else:
    st.info('Upload a dataset to start.');st.stop()
try:
    raw = load_data(data_bytes,filename)
    df,X = preprocess(raw,year,cap_outliers)
except Exception as exc:
    st.error(f'Dataset validation failed: {exc}');st.stop()

with st.spinner('Training and evaluating all five models...'):
    try:
        models = {
            'K-Means':KMeans(n_clusters=k,random_state=42,n_init=20),
            'Agglomerative':AgglomerativeClustering(n_clusters=k,linkage='ward'),
            'Gaussian Mixture Model':GaussianMixture(n_components=k,covariance_type='full',random_state=42,n_init=10),
            'BIRCH':Birch(threshold=0.8,n_clusters=k),
            'DBSCAN':DBSCAN(eps=eps,min_samples=min_samples)
        }
        labels = {name:model.fit_predict(X) for name,model in models.items()}
        scores = pd.DataFrame([evaluate(name,X,lab,name=='DBSCAN') for name,lab in labels.items()])
    except Exception as exc:
        st.error(f'Model fitting failed: {exc}');st.stop()

overview, modeling, evaluation, segments, downloads = st.tabs(['Dataset overview','Five models','Model evaluation','Customer segments','Download'])
with overview:
    a,b,c = st.columns(3)
    a.metric('Customers',f'{len(df):,}');b.metric('Original columns',raw.shape[1]);c.metric('Clustering features',len(FEATURES))
    st.subheader('Raw dataset preview');st.dataframe(raw.head(20),use_container_width=True)
    st.subheader('Missing values in original data');st.dataframe(raw.isna().sum().rename('Missing values').to_frame(),use_container_width=True)
    st.subheader('Feature distributions')
    choice = st.selectbox('Choose feature',FEATURES)
    st.plotly_chart(px.histogram(df,x=choice,nbins=35,title=f'Distribution of {choice}'),use_container_width=True)
with modeling:
    st.markdown('Each model is fitted to the same standardized 13-feature matrix. DBSCAN can label outliers as **-1**.')
    descriptions = {'K-Means':'Centroid-based clustering that minimizes within-cluster distances.','Agglomerative':'Bottom-up hierarchical clustering using Ward linkage.','Gaussian Mixture Model':'Probabilistic clustering with full covariance matrices.','BIRCH':'Builds compact subclusters before final clustering.','DBSCAN':'Density-based clustering; does not require a fixed cluster count.'}
    for idx,(name,lab) in enumerate(labels.items(),1):
        with st.expander(f'Model {idx}: {name}',expanded=(idx==1)):
            st.write(descriptions[name]);st.dataframe(pd.Series(lab).value_counts().sort_index().rename_axis('Cluster').rename('Customers').to_frame(),use_container_width=True)
with evaluation:
    st.subheader('Evaluation of all five models')
    st.caption('Higher Silhouette and Calinski-Harabasz are better; lower Davies-Bouldin is better. DBSCAN noise is excluded from its metrics and counted separately.')
    st.dataframe(scores.style.format({'Silhouette Score':'{:.4f}','Davies-Bouldin Index':'{:.4f}','Calinski-Harabasz Index':'{:.2f}'},na_rep='N/A'),use_container_width=True)
    for metric in ['Silhouette Score','Davies-Bouldin Index','Calinski-Harabasz Index']:
        st.plotly_chart(px.bar(scores,x='Model',y=metric,title=metric),use_container_width=True)
    valid=scores.dropna(subset=['Silhouette Score'])
    if len(valid):
        best=valid.loc[valid['Silhouette Score'].idxmax()]
        st.success(f"Highest Silhouette Score: {best['Model']} ({best['Silhouette Score']:.4f}). Confirm with other metrics and business interpretation before final selection.")
    else:st.warning('No model produced enough valid clusters to compute the internal metrics.')
with segments:
    selected=st.selectbox('Choose clustering model for profiling',list(labels))
    prof=df.copy();prof['Cluster']=labels[selected]
    st.subheader('Cluster size')
    sizes=prof['Cluster'].value_counts().sort_index().reset_index();sizes.columns=['Cluster','Customers'];sizes['Cluster']=sizes['Cluster'].astype(str)
    st.plotly_chart(px.bar(sizes,x='Cluster',y='Customers',title=f'{selected} cluster sizes'),use_container_width=True)
    st.subheader('Segment profiles (original units, after optional outlier capping)')
    st.dataframe(prof.groupby('Cluster')[FEATURES].mean().round(2),use_container_width=True)
    st.subheader('PCA 2D projection')
    coords=PCA(n_components=2).fit_transform(X)
    scatter=pd.DataFrame({'PC1':coords[:,0],'PC2':coords[:,1],'Cluster':pd.Series(labels[selected]).astype(str)})
    st.plotly_chart(px.scatter(scatter,x='PC1',y='PC2',color='Cluster',title=f'PCA: {selected}',opacity=0.65),use_container_width=True)
    st.caption('PCA is used for visualization only. Models are trained on all 13 standardized features. Cluster IDs are arbitrary, not predefined customer value labels.')
with downloads:
    output=df.copy()
    for name,lab in labels.items():output['Cluster_'+name.replace(' ','_')]=lab
    st.download_button('Download customer assignments (CSV)',output.to_csv(index=False).encode('utf-8'),'customer_segments.csv','text/csv')
    st.download_button('Download evaluation results (CSV)',scores.to_csv(index=False).encode('utf-8'),'model_evaluation.csv','text/csv')
    st.info('For deployment, push this folder to GitHub and set app.py as the Streamlit entrypoint.')
