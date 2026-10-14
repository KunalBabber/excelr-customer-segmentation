
"""
EXCELR CUSTOMER SEGMENTATION PROJECT
====================================

Project Type:
    Unsupervised Machine Learning

Application:
    Streamlit Interactive Dashboard

Project Objective:
    Group customers into meaningful segments based on
    their income, spending habits, purchasing behavior,
    campaign responses, and demographic information.

Machine Learning Models:
    1. K-Means Clustering
    2. Agglomerative Hierarchical Clustering
    3. Gaussian Mixture Model (GMM)
    4. BIRCH Clustering
    5. DBSCAN Clustering

Evaluation Metrics:
    1. Silhouette Score
    2. Davies-Bouldin Index
    3. Calinski-Harabasz Index

Visualization:
    Plotly Charts and Principal Component Analysis (PCA)
"""


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

# io allows reading uploaded files directly from memory.
import io

# NumPy supports numerical calculations and array operations.
import numpy as np

# Pandas is used for data cleaning and analysis.
import pandas as pd

# Streamlit creates the interactive web application.
import streamlit as st

# Plotly Express creates interactive charts.
import plotly.express as px

# StandardScaler standardizes numerical features.
from sklearn.preprocessing import StandardScaler

# Import four clustering algorithms.
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    Birch,
    DBSCAN
)

# Import the probabilistic clustering algorithm.
from sklearn.mixture import GaussianMixture

# Import clustering evaluation metrics.
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# PCA reduces data dimensions for visualization.
from sklearn.decomposition import PCA


# ============================================================
# STEP 2: CONFIGURE THE STREAMLIT DASHBOARD
# ============================================================

# Set the browser tab title, icon, and page layout.
st.set_page_config(
    page_title="ExcelR | Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# Display the application title.
st.title("Customer Segmentation | ExcelR")

# Display a short project description.
st.caption(
    "Five unsupervised clustering models · "
    "evaluation · segment profiling · PCA"
)


# ============================================================
# STEP 3: DEFINE FEATURES USED FOR CLUSTERING
# ============================================================

# Select 13 customer attributes for model training.
#
# These features describe:
# - Customer demographics
# - Customer income
# - Spending behavior
# - Purchase frequency
# - Marketing campaign response
# - Customer complaints

FEATURES = [
    "Income",
    "Age",
    "Children",
    "Recency",
    "TotalSpend",
    "TotalPurchases",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    "TotalCampaignAccepted",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "Complain"
]


# ============================================================
# STEP 4: DEFINE PRODUCT SPENDING COLUMNS
# ============================================================

# These columns represent spending across six
# different product categories.
#
# Their sum will create the TotalSpend feature.

SPEND = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]


# ============================================================
# STEP 5: DEFINE PURCHASE CHANNEL COLUMNS
# ============================================================

# These columns represent customer purchases
# through different sales channels.
#
# Their sum creates TotalPurchases.

PURCHASES = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]


# ============================================================
# STEP 6: DEFINE MARKETING CAMPAIGN COLUMNS
# ============================================================

# These columns indicate whether customers
# accepted previous marketing campaigns.
#
# Response represents the latest campaign response.
#
# Their sum creates TotalCampaignAccepted.

CAMPAIGNS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response"
]


# ============================================================
# STEP 7: DEFINE REQUIRED DATASET COLUMNS
# ============================================================

# These columns must exist in the uploaded dataset.
#
# Engineered features such as Age and TotalSpend
# will be calculated later.

REQUIRED = [
    "Year_Birth",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    "Complain"
] + SPEND + PURCHASES + CAMPAIGNS


# ============================================================
# STEP 8: CREATE DATASET LOADING FUNCTION
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(file_bytes, filename):
    """
    Load customer data from a CSV or Excel file.

    Parameters:
        file_bytes:
            Uploaded file content in bytes.

        filename:
            Name of the uploaded dataset.

    Returns:
        pandas.DataFrame containing customer records.

    Caching:
        Streamlit stores the loaded result to avoid
        reading the same file repeatedly.
    """

    # Check whether the uploaded file is a CSV file.
    if filename.lower().endswith(".csv"):

        # Read CSV data from memory.
        return pd.read_csv(
            io.BytesIO(file_bytes)
        )

    # Otherwise, read the uploaded Excel file.
    return pd.read_excel(
        io.BytesIO(file_bytes)
    )


# ============================================================
# STEP 9: CREATE DATA PREPROCESSING FUNCTION
# ============================================================

def preprocess(raw, year, cap_outliers):
    """
    Clean and prepare customer data for clustering.

    Main Operations:
        1. Validate required columns.
        2. Convert columns into numeric format.
        3. Calculate customer age.
        4. Handle missing values.
        5. Perform feature engineering.
        6. Apply optional IQR outlier capping.
        7. Standardize selected features.

    Returns:
        df:
            Cleaned customer dataset.

        X:
            Standardized feature matrix used by
            the five clustering algorithms.
    """

    # --------------------------------------------------------
    # STEP 9.1: COPY THE ORIGINAL DATASET
    # --------------------------------------------------------

    # Create a copy to protect the original dataset.
    df = raw.copy()


    # --------------------------------------------------------
    # STEP 9.2: VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    # Identify columns missing from the dataset.
    missing = [
        c for c in REQUIRED
        if c not in df.columns
    ]

    # Stop execution if required columns are missing.
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


    # --------------------------------------------------------
    # STEP 9.3: CONVERT COLUMNS TO NUMERIC FORMAT
    # --------------------------------------------------------

    # Convert required columns into numerical values.
    #
    # errors="coerce" converts invalid values into NaN.

    for c in set(REQUIRED):
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )


    # --------------------------------------------------------
    # STEP 9.4: FEATURE ENGINEERING - CUSTOMER AGE
    # --------------------------------------------------------

    # Calculate customer age from birth year.
    #
    # Formula:
    # Age = Reference Year - Year_Birth

    df["Age"] = year - df["Year_Birth"]

    # Treat unrealistic ages as missing values.
    #
    # Valid age range:
    # 18 to 100 years.

    df.loc[
        (df["Age"] < 18) | (df["Age"] > 100),
        "Age"
    ] = np.nan


    # --------------------------------------------------------
    # STEP 9.5: HANDLE MISSING INCOME AND AGE
    # --------------------------------------------------------

    # Fill missing Income values using median income.
    df["Income"] = df["Income"].fillna(
        df["Income"].median()
    )

    # Fill missing Age values using median age.
    df["Age"] = df["Age"].fillna(
        df["Age"].median()
    )


    # --------------------------------------------------------
    # STEP 9.6: FEATURE ENGINEERING - TOTAL SPENDING
    # --------------------------------------------------------

    # Calculate total customer spending.
    #
    # TotalSpend =
    # Wines + Fruits + Meat + Fish + Sweets + Gold

    df["TotalSpend"] = df[SPEND].sum(
        axis=1
    )


    # --------------------------------------------------------
    # STEP 9.7: FEATURE ENGINEERING - TOTAL PURCHASES
    # --------------------------------------------------------

    # Calculate total purchases through:
    # - Website
    # - Catalog
    # - Physical stores

    df["TotalPurchases"] = df[PURCHASES].sum(
        axis=1
    )


    # --------------------------------------------------------
    # STEP 9.8: FEATURE ENGINEERING - CHILDREN
    # --------------------------------------------------------

    # Calculate the total number of children
    # and teenagers in each household.

    df["Children"] = df[
        ["Kidhome", "Teenhome"]
    ].sum(axis=1)


    # --------------------------------------------------------
    # STEP 9.9: FEATURE ENGINEERING - CAMPAIGN ACCEPTANCE
    # --------------------------------------------------------

    # Calculate the total number of accepted campaigns.
    #
    # The result includes five previous campaigns
    # and the latest campaign response.

    df["TotalCampaignAccepted"] = df[
        CAMPAIGNS
    ].sum(axis=1)


    # --------------------------------------------------------
    # STEP 9.10: HANDLE REMAINING MISSING VALUES
    # --------------------------------------------------------

    # Fill missing values in all selected features
    # using their respective median values.

    for col in FEATURES:

        df[col] = df[col].fillna(
            df[col].median()
        )

        # Stop if a feature has no usable numerical values.
        if df[col].isna().any():
            raise ValueError(
                f"Column {col} has no usable numerical values."
            )


    # --------------------------------------------------------
    # STEP 9.11: OPTIONAL OUTLIER CAPPING USING IQR
    # --------------------------------------------------------

    # IQR stands for Interquartile Range.
    #
    # Q1 = 25th percentile
    # Q3 = 75th percentile
    #
    # IQR = Q3 - Q1
    #
    # Lower Limit = Q1 - 1.5 * IQR
    # Upper Limit = Q3 + 1.5 * IQR
    #
    # Values outside these limits are capped.
    #
    # This method limits extreme values instead
    # of deleting customer records.

    if cap_outliers:

        for col in FEATURES:

            # Calculate the first and third quartiles.
            q1, q3 = df[col].quantile(
                [0.25, 0.75]
            )

            # Calculate the interquartile range.
            iqr = q3 - q1

            # Calculate the lower and upper boundaries.
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            # Cap values outside the IQR boundaries.
            df[col] = df[col].clip(
                lower_limit,
                upper_limit
            )


    # --------------------------------------------------------
    # STEP 9.12: STANDARDIZE FEATURES
    # --------------------------------------------------------

    # StandardScaler transforms features so that
    # each feature has approximately:
    #
    # Mean = 0
    # Standard Deviation = 1
    #
    # Formula:
    # Z = (X - Mean) / Standard Deviation
    #
    # Scaling is important because features such
    # as Income and Children have different ranges.

    scaler = StandardScaler()

    # Fit and transform the 13 selected features.
    X = scaler.fit_transform(
        df[FEATURES]
    )


    # --------------------------------------------------------
    # STEP 9.13: RETURN PREPROCESSED DATA
    # --------------------------------------------------------

    # df contains cleaned customer information.
    #
    # X contains standardized numerical features
    # ready for clustering model training.

    return df, X


# ============================================================
# STEP 10: CREATE MODEL EVALUATION FUNCTION
# ============================================================

def evaluate(name, X, labels, exclude_noise=False):
    """
    Evaluate clustering performance.

    Parameters:
        name:
            Name of the clustering model.

        X:
            Standardized customer feature matrix.

        labels:
            Predicted cluster assignments.

        exclude_noise:
            If True, exclude DBSCAN noise observations.

    Evaluation Metrics:
        1. Silhouette Score
        2. Davies-Bouldin Index
        3. Calinski-Harabasz Index

    Returns:
        Dictionary containing evaluation results.
    """

    # Convert cluster labels into a NumPy array.
    labels = np.asarray(labels)


    # --------------------------------------------------------
    # STEP 10.1: COUNT NOISE POINTS
    # --------------------------------------------------------

    # DBSCAN labels noise observations as -1.
    #
    # Count how many customers were classified as noise.

    noise = int(
        (labels == -1).sum()
    )


    # --------------------------------------------------------
    # STEP 10.2: FILTER NOISE OBSERVATIONS
    # --------------------------------------------------------

    # Exclude noise points only when requested.
    #
    # For DBSCAN:
    #     Exclude label -1.
    #
    # For other models:
    #     Include all customers.

    mask = (
        labels != -1
        if exclude_noise
        else np.ones(len(labels), dtype=bool)
    )

    # Select valid cluster labels.
    clean_labels = labels[mask]

    # Select corresponding customer features.
    clean_X = X[mask]


    # --------------------------------------------------------
    # STEP 10.3: COUNT VALID CLUSTERS
    # --------------------------------------------------------

    # Count the number of unique cluster labels.
    clusters = len(
        np.unique(clean_labels)
    )


    # --------------------------------------------------------
    # STEP 10.4: INITIALIZE EVALUATION RESULTS
    # --------------------------------------------------------

    # Use NaN as the default value for metrics.
    #
    # This handles situations where the clustering
    # result does not contain enough valid clusters.

    result = {
        "Model": name,
        "Clusters": clusters,
        "Noise Points": noise,
        "Silhouette Score": np.nan,
        "Davies-Bouldin Index": np.nan,
        "Calinski-Harabasz Index": np.nan
    }


    # --------------------------------------------------------
    # STEP 10.5: VALIDATE THE NUMBER OF CLUSTERS
    # --------------------------------------------------------

    # Evaluation metrics require at least two clusters.
    #
    # The number of clusters must also be smaller
    # than the number of evaluated samples.

    if 2 <= clusters < len(clean_labels):


        # ----------------------------------------------------
        # METRIC 1: SILHOUETTE SCORE
        # ----------------------------------------------------

        # Measures how well customers fit within
        # their assigned clusters.
        #
        # Range:
        # -1 to +1
        #
        # Higher values generally indicate better
        # cohesion and separation.

        silhouette = silhouette_score(
            clean_X,
            clean_labels
        )


        # ----------------------------------------------------
        # METRIC 2: DAVIES-BOULDIN INDEX
        # ----------------------------------------------------

        # Measures cluster similarity based on
        # within-cluster dispersion and separation.
        #
        # Lower values indicate better clustering.

        dbi = davies_bouldin_score(
            clean_X,
            clean_labels
        )


        # ----------------------------------------------------
        # METRIC 3: CALINSKI-HARABASZ INDEX
        # ----------------------------------------------------

        # Measures between-cluster separation
        # relative to within-cluster dispersion.
        #
        # Higher values generally indicate better
        # clustering structure.

        chi = calinski_harabasz_score(
            clean_X,
            clean_labels
        )


        # ----------------------------------------------------
        # STEP 10.6: STORE EVALUATION METRICS
        # --------------------------------------------------------

        result.update({
            "Silhouette Score": silhouette,
            "Davies-Bouldin Index": dbi,
            "Calinski-Harabasz Index": chi
        })


    # Return the evaluation results for this model.
    return result


# ============================================================
# STEP 11: CREATE SIDEBAR SETTINGS
# ============================================================

# The sidebar allows users to control dataset selection,
# clustering parameters, and preprocessing options.

st.sidebar.header(
    "Dataset & settings"
)


# ------------------------------------------------------------
# STEP 11.1: DATASET UPLOAD
# ------------------------------------------------------------

# Allow users to upload Excel or CSV datasets.

upload = st.sidebar.file_uploader(
    "Upload marketing campaign dataset",
    type=["xlsx", "csv"]
)


# ------------------------------------------------------------
# STEP 11.2: DEFAULT DATASET OPTION
# ------------------------------------------------------------

# If enabled, use the ExcelR dataset stored locally.

use_sample = st.sidebar.checkbox(
    "Use bundled ExcelR dataset",
    value=True
)


# ------------------------------------------------------------
# STEP 11.3: REFERENCE YEAR
# ------------------------------------------------------------

# Allow the user to select the year used
# for calculating customer age.

year = st.sidebar.number_input(
    "Reference year for customer age",
    min_value=2015,
    max_value=2100,
    value=2026
)


# ------------------------------------------------------------
# STEP 11.4: NUMBER OF CLUSTERS
# ------------------------------------------------------------

# Set the number of clusters for:
# - K-Means
# - Agglomerative
# - Gaussian Mixture Model
# - BIRCH

k = st.sidebar.slider(
    "Clusters for K-Means / GMM / Ward / BIRCH",
    2,
    8,
    3
)


# ------------------------------------------------------------
# STEP 11.5: DBSCAN PARAMETERS
# ------------------------------------------------------------

# eps controls the neighborhood radius used by DBSCAN.

eps = st.sidebar.slider(
    "DBSCAN eps",
    0.2,
    5.0,
    1.8,
    0.1
)

# min_samples controls the minimum neighborhood
# sample count required for a core point.

min_samples = st.sidebar.slider(
    "DBSCAN min_samples",
    2,
    40,
    10
)


# ------------------------------------------------------------
# STEP 11.6: OUTLIER CAPPING OPTION
# ------------------------------------------------------------

# Enable or disable IQR-based outlier treatment.

cap_outliers = st.sidebar.checkbox(
    "Apply IQR outlier capping",
    value=True
)


# ============================================================
# STEP 12: LOAD THE SELECTED DATASET
# ============================================================

# If the user uploads a file, use that file.
if upload is not None:

    data_bytes = upload.getvalue()
    filename = upload.name

# Otherwise, use the bundled ExcelR dataset.
elif use_sample:

    try:

        # Open the default Excel dataset in binary mode.
        with open(
            "marketing_campaign.xlsx",
            "rb"
        ) as f:

            data_bytes = f.read()

        filename = "marketing_campaign.xlsx"

    # Handle missing default dataset.
    except FileNotFoundError:

        st.warning(
            "Bundled dataset missing. "
            "Upload an Excel/CSV file."
        )

        st.stop()

# Stop if no dataset is selected.
else:

    st.info(
        "Upload a dataset to start."
    )

    st.stop()


# ============================================================
# STEP 13: PREPROCESS THE CUSTOMER DATASET
# ============================================================

try:

    # Read the selected customer dataset.
    raw = load_data(
        data_bytes,
        filename
    )

    # Apply preprocessing and feature scaling.
    df, X = preprocess(
        raw,
        year,
        cap_outliers
    )

# Display an error if dataset processing fails.
except Exception as exc:

    st.error(
        f"Dataset validation failed: {exc}"
    )

    st.stop()


# ============================================================
# STEP 14: TRAIN FIVE CLUSTERING MODELS
# ============================================================

# Display a loading message during model training.
with st.spinner(
    "Training and evaluating all five models..."
):

    try:

        # ----------------------------------------------------
        # MODEL 1: K-MEANS CLUSTERING
        # ----------------------------------------------------

        # K-Means groups customers around cluster centers.
        #
        # n_clusters=k:
        #     Number of customer segments.
        #
        # random_state=42:
        #     Makes initialization reproducible.
        #
        # n_init=20:
        #     Runs 20 initializations and retains
        #     the best result by inertia.

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )


        # ----------------------------------------------------
        # MODEL 2: AGGLOMERATIVE CLUSTERING
        # ----------------------------------------------------

        # Agglomerative clustering starts with
        # individual customers as separate clusters.
        #
        # It repeatedly merges clusters until
        # the required number of groups remains.
        #
        # Ward linkage minimizes the increase
        # in within-cluster variance.

        agglomerative = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward"
        )


        # ----------------------------------------------------
        # MODEL 3: GAUSSIAN MIXTURE MODEL
        # ----------------------------------------------------

        # GMM uses Gaussian probability distributions
        # to identify customer groups.
        #
        # n_components=k:
        #     Number of Gaussian components.
        #
        # covariance_type="full":
        #     Each component has its own covariance matrix.
        #
        # n_init=10:
        #     Uses multiple initializations to find
        #     a solution with a high likelihood bound.

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            n_init=10
        )


        # ----------------------------------------------------
        # MODEL 4: BIRCH CLUSTERING
        # ----------------------------------------------------

        # BIRCH stands for:
        # Balanced Iterative Reducing and Clustering
        # using Hierarchies.
        #
        # It creates compact subclusters using
        # a clustering feature tree.
        #
        # threshold=0.8:
        #     Controls the subcluster radius threshold.
        #
        # n_clusters=k:
        #     Requested final cluster count.

        birch = Birch(
            threshold=0.8,
            n_clusters=k
        )


        # ----------------------------------------------------
        # MODEL 5: DBSCAN CLUSTERING
        # ----------------------------------------------------

        # DBSCAN stands for:
        # Density-Based Spatial Clustering of
        # Applications with Noise.
        #
        # DBSCAN identifies dense customer groups.
        #
        # It does not require a fixed cluster count.
        #
        # eps:
        #     Neighborhood distance threshold.
        #
        # min_samples:
        #     Minimum neighborhood sample count
        #     for a core point.
        #
        # Noise observations are labeled -1.

        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )


        # ----------------------------------------------------
        # STEP 14.1: STORE ALL MODELS
        # ----------------------------------------------------

        models = {
            "K-Means": kmeans,
            "Agglomerative": agglomerative,
            "Gaussian Mixture Model": gmm,
            "BIRCH": birch,
            "DBSCAN": dbscan
        }


        # ----------------------------------------------------
        # STEP 14.2: TRAIN ALL FIVE MODELS
        # ----------------------------------------------------

        # fit_predict():
        #     Fits each model and assigns cluster labels.
        #
        # The same standardized features are used
        # for all five clustering algorithms.

        labels = {
            name: model.fit_predict(X)
            for name, model in models.items()
        }


        # ----------------------------------------------------
        # STEP 14.3: EVALUATE ALL FIVE MODELS
        # ----------------------------------------------------

        # Calculate evaluation metrics for each model.
        #
        # DBSCAN noise observations are excluded
        # from its evaluation metrics.

        scores = pd.DataFrame([
            evaluate(
                name,
                X,
                lab,
                name == "DBSCAN"
            )
            for name, lab in labels.items()
        ])


    # Handle model training errors.
    except Exception as exc:

        st.error(
            f"Model fitting failed: {exc}"
        )

        st.stop()


# ============================================================
# STEP 15: CREATE DASHBOARD NAVIGATION TABS
# ============================================================

# Divide the application into five main sections.

overview, modeling, evaluation, segments, downloads = st.tabs([
    "Dataset overview",
    "Five models",
    "Model evaluation",
    "Customer segments",
    "Download"
])


# ============================================================
# STEP 16: DATASET OVERVIEW AND EDA
# ============================================================

with overview:

    # --------------------------------------------------------
    # STEP 16.1: DISPLAY DATASET STATISTICS
    # --------------------------------------------------------

    # Create three columns for summary metrics.
    a, b, c = st.columns(3)

    # Display the total number of customers.
    a.metric(
        "Customers",
        f"{len(df):,}"
    )

    # Display the original dataset column count.
    b.metric(
        "Original columns",
        raw.shape[1]
    )

    # Display the number of features used for clustering.
    c.metric(
        "Clustering features",
        len(FEATURES)
    )


    # --------------------------------------------------------
    # STEP 16.2: DISPLAY RAW DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "Raw dataset preview"
    )

    # Display the first 20 customer records.
    st.dataframe(
        raw.head(20),
        use_container_width=True
    )


    # --------------------------------------------------------
    # STEP 16.3: DISPLAY MISSING VALUES
    # --------------------------------------------------------

    st.subheader(
        "Missing values in original data"
    )

    # Count missing values in each original column.
    missing_values = (
        raw.isna()
        .sum()
        .rename("Missing values")
        .to_frame()
    )

    st.dataframe(
        missing_values,
        use_container_width=True
    )


    # --------------------------------------------------------
    # STEP 16.4: FEATURE DISTRIBUTION ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "Feature distributions"
    )

    # Allow the user to select a feature for visualization.
    choice = st.selectbox(
        "Choose feature",
        FEATURES
    )

    # Create a histogram for the selected feature.
    #
    # X-axis:
    #     Selected customer feature.
    #
    # Y-axis:
    #     Number of customers.

    distribution_chart = px.histogram(
        df,
        x=choice,
        nbins=35,
        title=f"Distribution of {choice}"
    )

    # Display the interactive histogram.
    st.plotly_chart(
        distribution_chart,
        use_container_width=True
    )


# ============================================================
# STEP 17: FIVE CLUSTERING MODEL RESULTS
# ============================================================

with modeling:

    # Explain the modeling approach.
    st.markdown(
        "Each model is fitted to the same standardized "
        "13-feature matrix. DBSCAN can label outliers as **-1**."
    )


    # --------------------------------------------------------
    # STEP 17.1: DEFINE MODEL DESCRIPTIONS
    # --------------------------------------------------------

    # These descriptions help explain each clustering method.

    descriptions = {
        "K-Means":
            "Centroid-based clustering that minimizes "
            "within-cluster distances.",

        "Agglomerative":
            "Bottom-up hierarchical clustering "
            "using Ward linkage.",

        "Gaussian Mixture Model":
            "Probabilistic clustering with "
            "full covariance matrices.",

        "BIRCH":
            "Builds compact subclusters "
            "before final clustering.",

        "DBSCAN":
            "Density-based clustering; does not "
            "require a fixed cluster count."
    }


    # --------------------------------------------------------
    # STEP 17.2: DISPLAY EACH MODEL
    # --------------------------------------------------------

    # Loop through all five clustering models.

    for idx, (name, lab) in enumerate(
        labels.items(),
        1
    ):

        # Create an expandable section for each model.
        with st.expander(
            f"Model {idx}: {name}",
            expanded=(idx == 1)
        ):

            # Display the model description.
            st.write(
                descriptions[name]
            )


            # ------------------------------------------------
            # STEP 17.3: COUNT CUSTOMERS IN EACH CLUSTER
            # ------------------------------------------------

            # Convert cluster labels into a Pandas Series.
            #
            # value_counts():
            #     Counts customers assigned to each cluster.
            #
            # sort_index():
            #     Sorts cluster labels numerically.
            #
            # DBSCAN label -1 represents noise.

            cluster_counts = (
                pd.Series(lab)
                .value_counts()
                .sort_index()
                .rename_axis("Cluster")
                .rename("Customers")
                .to_frame()
            )

            # Display the cluster distribution table.
            st.dataframe(
                cluster_counts,
                use_container_width=True
            )


# ============================================================
# STEP 18: MODEL EVALUATION DASHBOARD
# ============================================================

with evaluation:

    st.subheader(
        "Evaluation of all five models"
    )

    # Explain the interpretation of evaluation metrics.
    st.caption(
        "Higher Silhouette and Calinski-Harabasz are better; "
        "lower Davies-Bouldin is better. "
        "DBSCAN noise is excluded from its metrics "
        "and counted separately."
    )


    # --------------------------------------------------------
    # STEP 18.1: DISPLAY EVALUATION TABLE
    # --------------------------------------------------------

    # Format metric values for better readability.
    #
    # Undefined metric values are displayed as N/A.

    st.dataframe(
        scores.style.format(
            {
                "Silhouette Score": "{:.4f}",
                "Davies-Bouldin Index": "{:.4f}",
                "Calinski-Harabasz Index": "{:.2f}"
            },
            na_rep="N/A"
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # STEP 18.2: DISPLAY METRIC COMPARISON CHARTS
    # --------------------------------------------------------

    # Generate a separate bar chart for each metric.

    for metric in [
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Index"
    ]:

        # X-axis: Clustering model.
        # Y-axis: Evaluation metric value.

        metric_chart = px.bar(
            scores,
            x="Model",
            y=metric,
            title=metric
        )

        st.plotly_chart(
            metric_chart,
            use_container_width=True
        )


    # --------------------------------------------------------
    # STEP 18.3: IDENTIFY THE HIGHEST SILHOUETTE SCORE
    # --------------------------------------------------------

    # Remove models with undefined Silhouette Scores.
    valid = scores.dropna(
        subset=["Silhouette Score"]
    )

    if len(valid):

        # Identify the model with the highest score.
        best = valid.loc[
            valid["Silhouette Score"].idxmax()
        ]

        # Display the highest Silhouette Score.
        #
        # This is not automatically the best model
        # for every business objective.

        st.success(
            f"Highest Silhouette Score: "
            f"{best['Model']} "
            f"({best['Silhouette Score']:.4f}). "
            "Confirm with other metrics and business "
            "interpretation before final selection."
        )

    else:

        # Display a warning when metrics are undefined.
        st.warning(
            "No model produced enough valid clusters "
            "to compute the internal metrics."
        )


# ============================================================
# STEP 19: CUSTOMER SEGMENTATION AND PCA
# ============================================================

with segments:

    # --------------------------------------------------------
    # STEP 19.1: SELECT A CLUSTERING MODEL
    # --------------------------------------------------------

    # Allow the user to select any of the five models.
    selected = st.selectbox(
        "Choose clustering model for profiling",
        list(labels)
    )


    # --------------------------------------------------------
    # STEP 19.2: ASSIGN CUSTOMER CLUSTER LABELS
    # --------------------------------------------------------

    # Create a copy of the cleaned dataset.
    prof = df.copy()

    # Add the selected model's cluster assignments.
    prof["Cluster"] = labels[selected]


    # --------------------------------------------------------
    # STEP 19.3: DISPLAY CLUSTER SIZES
    # --------------------------------------------------------

    st.subheader(
        "Cluster size"
    )

    # Count customers in each cluster.
    sizes = (
        prof["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    # Rename the resulting columns.
    sizes.columns = [
        "Cluster",
        "Customers"
    ]

    # Convert cluster labels to strings for Plotly.
    sizes["Cluster"] = sizes["Cluster"].astype(str)

    # Create a bar chart of customer counts.
    cluster_chart = px.bar(
        sizes,
        x="Cluster",
        y="Customers",
        title=f"{selected} cluster sizes"
    )

    st.plotly_chart(
        cluster_chart,
        use_container_width=True
    )


    # --------------------------------------------------------
    # STEP 19.4: CUSTOMER SEGMENT PROFILING
    # --------------------------------------------------------

    st.subheader(
        "Segment profiles "
        "(original units, after optional outlier capping)"
    )

    # Group customers by their assigned cluster.
    #
    # Calculate the average of all 13 clustering features.
    #
    # This helps identify behavioral differences
    # between customer segments.

    segment_profiles = (
        prof.groupby("Cluster")[FEATURES]
        .mean()
        .round(2)
    )

    # Display the customer segment profile table.
    st.dataframe(
        segment_profiles,
        use_container_width=True
    )


    # --------------------------------------------------------
    # STEP 19.5: PRINCIPAL COMPONENT ANALYSIS (PCA)
    # --------------------------------------------------------

    st.subheader(
        "PCA 2D projection"
    )

    # PCA reduces the 13-dimensional feature matrix
    # into two principal components.
    #
    # PCA is used only for visualization.
    #
    # All clustering models were trained on the
    # original 13 standardized features.

    pca = PCA(
        n_components=2
    )

    # Transform standardized features into two dimensions.
    coords = pca.fit_transform(X)


    # --------------------------------------------------------
    # STEP 19.6: CREATE PCA SCATTER DATAFRAME
    # --------------------------------------------------------

    # Create a DataFrame containing:
    # - Principal Component 1
    # - Principal Component 2
    # - Customer cluster labels

    scatter = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Cluster": pd.Series(
            labels[selected]
        ).astype(str)
    })


    # --------------------------------------------------------
    # STEP 19.7: DISPLAY PCA SCATTER PLOT
    # --------------------------------------------------------

    # Each point represents one customer.
    #
    # Colors indicate different cluster assignments.
    #
    # Cluster numbers are arbitrary identifiers.

    pca_chart = px.scatter(
        scatter,
        x="PC1",
        y="PC2",
        color="Cluster",
        title=f"PCA: {selected}",
        opacity=0.65
    )

    st.plotly_chart(
        pca_chart,
        use_container_width=True
    )

    # Explain how the PCA plot should be interpreted.
    st.caption(
        "PCA is used for visualization only. "
        "Models are trained on all 13 standardized features. "
        "Cluster IDs are arbitrary, not predefined "
        "customer value labels."
    )


# ============================================================
# STEP 20: DOWNLOAD CUSTOMER SEGMENTATION RESULTS
# ============================================================

with downloads:

    # --------------------------------------------------------
    # STEP 20.1: CREATE FINAL OUTPUT DATASET
    # --------------------------------------------------------

    # Create a copy of the cleaned customer dataset.
    output = df.copy()


    # --------------------------------------------------------
    # STEP 20.2: ADD CLUSTER LABELS FROM ALL FIVE MODELS
    # --------------------------------------------------------

    # Add a separate cluster column for every model.
    #
    # Example:
    # Cluster_K-Means
    # Cluster_Agglomerative
    # Cluster_Gaussian_Mixture_Model
    # Cluster_BIRCH
    # Cluster_DBSCAN

    for name, lab in labels.items():

        output[
            "Cluster_" + name.replace(" ", "_")
        ] = lab


    # --------------------------------------------------------
    # STEP 20.3: DOWNLOAD CUSTOMER ASSIGNMENTS
    # --------------------------------------------------------

    # Convert customer segmentation results to CSV.
    #
    # index=False excludes the Pandas index.

    customer_csv = (
        output.to_csv(index=False)
        .encode("utf-8")
    )

    # Create a download button for customer segments.
    st.download_button(
        "Download customer assignments (CSV)",
        customer_csv,
        "customer_segments.csv",
        "text/csv"
    )


    # --------------------------------------------------------
    # STEP 20.4: DOWNLOAD MODEL EVALUATION RESULTS
    # --------------------------------------------------------

    # Convert model evaluation results to CSV.
    evaluation_csv = (
        scores.to_csv(index=False)
        .encode("utf-8")
    )

    # Create a download button for evaluation results.
    st.download_button(
        "Download evaluation results (CSV)",
        evaluation_csv,
        "model_evaluation.csv",
        "text/csv"
    )


    # --------------------------------------------------------
    # STEP 20.5: DISPLAY DEPLOYMENT INFORMATION
    # --------------------------------------------------------

    # Explain the basic Streamlit deployment process.
    st.info(
        "For deployment, push this folder to GitHub "
        "and set app.py as the Streamlit entrypoint."
    )
