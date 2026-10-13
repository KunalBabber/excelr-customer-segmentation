"""Preprocessing aligned to the uploaded ExcelR notebook, Sections 4–7."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
FEATURES = ["Income","Age","Children","Recency","TotalSpend","TotalPurchases","NumDealsPurchases","NumWebVisitsMonth","TotalCampaignAccepted","NumWebPurchases","NumCatalogPurchases","NumStorePurchases","Complain"]
SPENDING = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
PURCHASES = ["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]
CAMPAIGNS = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"]
def prepare(raw):
    df=raw.copy()
    required=set(FEATURES)-{"Age","Children","TotalSpend","TotalPurchases","TotalCampaignAccepted"}
    required |= set(SPENDING+PURCHASES+CAMPAIGNS+["Year_Birth","Kidhome","Teenhome"])
    missing=sorted(required-set(df.columns))
    if missing: raise ValueError("Dataset missing required columns: "+", ".join(missing))
    df=df.drop_duplicates().reset_index(drop=True)
    for col in required: df[col]=pd.to_numeric(df[col],errors="coerce")
    df["Age"]=2026-df["Year_Birth"]
    df.loc[(df["Age"]<18)|(df["Age"]>100),"Age"]=np.nan
    for col in ["Age","Income"]: df[col]=df[col].fillna(df[col].median())
    df["TotalSpend"]=df[SPENDING].sum(axis=1)
    df["TotalPurchases"]=df[PURCHASES].sum(axis=1)
    df["Children"]=df["Kidhome"]+df["Teenhome"]
    df["TotalCampaignAccepted"]=df[CAMPAIGNS].sum(axis=1)
    # Notebook applies IQR capping to every numerical column, then standardizes FEATURES.
    numeric=df.select_dtypes(include="number").columns
    for col in numeric:
        q1,q3=df[col].quantile([.25,.75]);iqr=q3-q1
        df[col]=df[col].clip(q1-1.5*iqr,q3+1.5*iqr)
    if df[FEATURES].isna().any().any():
        df[FEATURES]=df[FEATURES].fillna(df[FEATURES].median())
    if df[FEATURES].isna().any().any():raise ValueError("Selected features still contain missing values")
    profile=df.copy()
    scaled=StandardScaler().fit_transform(df[FEATURES])
    return profile,pd.DataFrame(scaled,columns=FEATURES,index=df.index)
