
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# PAGE SETTINGS
st.set_page_config(page_title="App User Segmentation", page_icon="📊", layout="wide")

DATA_FILE = "app_user_behavior_preprocess.csv"
MODEL_FILE = "kmeans_model.joblib"
SCALER_FILE = "scaler.joblib"
K = 4

FEATURES = [
    "sessions_per_week", "avg_session_duration_min",
    "daily_active_minutes", "feature_clicks_per_session",
    "notifications_opened_per_week", "in_app_search_count",
    "pages_viewed_per_session", "days_since_last_login",
    "content_downloads", "engagement_score"
]

PDF_ACTIONS = {
    "High Engagement Users": "Loyalty programs, premium offers and early access to new features.",
    "Moderate Engagement Users": "Personalised engagement: tailored notifications, recommendations and feature tips.",
    "Low Engagement / At-Risk Users": "Retention campaigns: offers, reminders and improved onboarding.",
    "Occasional Users": "Re-engagement nudges and simple prompts to build a usage habit."
}


# LOAD DATA
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


# BUILD MODEL
@st.cache_resource
def build_model(df):
    X = df[FEATURES]

    if os.path.exists(SCALER_FILE) and os.path.exists(MODEL_FILE):
        scaler = joblib.load(SCALER_FILE)
        model = joblib.load(MODEL_FILE)
    else:
        scaler = StandardScaler().fit(X)
        model = KMeans(n_clusters=K, random_state=42, n_init=10)
        model.fit(scaler.transform(X))

    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)

    return scaler, model, X_scaled, labels


# FIND OPTIMAL K
@st.cache_data
def k_selection(X):
    ks = range(1, 11)
    inertia = [
        KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
        for k in ks
    ]

    sil = {}
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        sil[k] = silhouette_score(X, labels, sample_size=10000, random_state=42)

    return list(ks), inertia, sil


# PCA
@st.cache_data
def pca_2d(X):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_


# NAME CLUSTERS
def name_clusters(df, labels):
    z = (df[FEATURES].groupby(labels).mean() - df[FEATURES].mean()) / df[FEATURES].std()

    level = (
        z["sessions_per_week"] +
        z["avg_session_duration_min"] +
        z["daily_active_minutes"] +
        z["pages_viewed_per_session"] +
        z["engagement_score"] -
        z["days_since_last_login"]
    )

    interaction = (
        z["feature_clicks_per_session"] +
        z["in_app_search_count"] +
        z["content_downloads"]
    )

    order = level.sort_values(ascending=False).index.tolist()

    names = {
        order[0]: "High Engagement Users",
        order[-1]: "Low Engagement / At-Risk Users"
    }

    middle = sorted(order[1:-1], key=lambda c: interaction[c])
    names[middle[0]] = "Occasional Users"
    names[middle[1]] = "Moderate Engagement Users"

    actions = {c: PDF_ACTIONS[name] for c, name in names.items()}

    return z, names, actions


# SIDEBAR AND DATA
st.sidebar.title("📊 App User Segmentation")

if os.path.exists(DATA_FILE):
    df = load_data(DATA_FILE)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is None:
        st.info(f"Place {DATA_FILE} next to app.py or upload it in the sidebar.")
        st.stop()

    df = load_data(uploaded_file)


# CHECK COLUMNS
required = FEATURES + ["user_id", "churn_risk_score"]
missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()


# MODEL AND CLUSTER INFORMATION
scaler, model, X_scaled, labels = build_model(df)
z, names, actions = name_clusters(df, labels)

dfc = df.copy()
dfc["Cluster"] = labels
dfc["Cluster_Name"] = dfc["Cluster"].map(names)

label_for = lambda c: f"{c} - {names[c]}"


# NAVIGATION
pages = [
    "Overview",
    "Optimal K (Elbow)",
    "Cluster Profiles",
    "PCA View",
    "User Explorer",
    "Business Actions",
    "Predict New User"
]

page = st.sidebar.radio("Go to", pages)
st.sidebar.caption("K-Means, k=4, StandardScaler, 10 behavioural features")


# 1. OVERVIEW
if page == "Overview":

    st.title("App User Behavior Segmentation")
    st.write("Unsupervised segmentation of app users using K-Means.")

    sil = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total users", f"{len(dfc):,}")
    c2.metric("Clusters", K)
    c3.metric("Features used", len(FEATURES))
    c4.metric("Silhouette score", f"{sil:.3f}")

   

    left, right = st.columns(2)
    counts = dfc["Cluster_Name"].value_counts()

    with left:
        st.subheader("Users per cluster")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#4C78A8")
        ax.set_xlabel("Users")
        ax.set_ylabel("")

        for i, value in enumerate(counts.values):
            ax.text(value, i, f" {value:,}", va="center")

        st.pyplot(fig)

    with right:
        st.subheader("Average churn risk per cluster")
        churn = dfc.groupby("Cluster_Name")["churn_risk_score"].mean().loc[counts.index]

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=churn.values, y=churn.index, ax=ax, color="#E45756")
        ax.set_xlabel("Mean churn_risk_score")
        ax.set_ylabel("")
        st.pyplot(fig)

    st.subheader("Cluster summary")

    profile = dfc.groupby("Cluster_Name").agg(
        users=("user_id", "count"),
        sessions=("sessions_per_week", "mean"),
        engagement=("engagement_score", "mean"),
        churn_risk=("churn_risk_score", "mean"),
        last_login=("days_since_last_login", "mean")
    ).round(3)

    st.dataframe(profile, use_container_width=True)


# 2. OPTIMAL K
elif page == "Optimal K (Elbow)":

    st.title("Optimal number of clusters")

    ks, inertia, sil = k_selection(X_scaled)
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Elbow method")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, inertia, marker="o")
        ax.axvline(K, color="red", ls="--", label=f"k = {K}")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Inertia")
        ax.legend()
        st.pyplot(fig)

    with c2:
        st.subheader("Silhouette score (10,000-row sample)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(sil.keys()), list(sil.values()), marker="o", color="green")
        ax.axvline(K, color="red", ls="--", label=f"k = {K}")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Silhouette")
        ax.legend()
        st.pyplot(fig)

    st.dataframe(
        pd.DataFrame({
            "k": list(sil.keys()),
            "silhouette": np.round(list(sil.values()), 4)
        }),
        hide_index=True
    )

    st.markdown(
        "**Why K-Means?** It scales to 50,000 rows, lets us choose a "
        "business-friendly number of segments, suits scaled numeric data, "
        "assigns every user to a cluster (DBSCAN left ~26% as noise) "
        "and gives easy-to-read centroids."
    )


# 3. CLUSTER PROFILES
elif page == "Cluster Profiles":

    st.title("Cluster profiles")
    st.subheader("What defines each cluster (z-score vs overall average)")

    zt = z.copy()
    zt.index = [label_for(c) for c in zt.index]

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(zt, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_xlabel("")
    plt.xticks(rotation=40, ha="right")
    st.pyplot(fig)

    st.caption(
        "Positive = above the overall average, negative = below. "
        "Values near 0 mean the cluster is not different on that feature."
    )

    st.subheader("Raw feature means per cluster")

    means = dfc.groupby("Cluster_Name")[FEATURES + ["churn_risk_score"]].mean().round(3)
    st.dataframe(means, use_container_width=True)

    feature = st.selectbox("Compare a feature across clusters", FEATURES + ["churn_risk_score"])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=dfc, x="Cluster_Name", y=feature, ax=ax, showfliers=False)
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    st.pyplot(fig)


# 4. PCA VIEW
elif page == "PCA View":

    st.title("PCA visualisation")

    pcs, variance = pca_2d(X_scaled)

    n = st.slider(
        "Points to plot",
        2000,
        len(pcs),
        min(15000, len(pcs)),
        step=1000
    )

    idx = np.random.RandomState(42).choice(len(pcs), n, replace=False)

    fig, ax = plt.subplots(figsize=(9, 6))

    for c in sorted(names):
        mask = labels[idx] == c
        ax.scatter(
            pcs[idx][mask, 0],
            pcs[idx][mask, 1],
            s=4,
            alpha=0.6,
            label=label_for(c)
        )

    ax.set_xlabel(f"PC1 ({variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%} variance)")
    ax.legend(markerscale=4)

    st.pyplot(fig)

    st.caption(
        f"The two components explain {variance.sum():.1%} of the variance; "
        "overlap here reflects the low silhouette score."
    )


# 5. USER EXPLORER
elif page == "User Explorer":

    st.title("Cluster-level user identification")

    choice = st.selectbox(
        "Cluster",
        ["All"] + [label_for(c) for c in sorted(names)]
    )

    view = dfc if choice == "All" else dfc[dfc["Cluster"] == int(choice.split(" - ")[0])]

    uid = st.text_input("Search by user_id")

    if uid.strip():
        view = view[view["user_id"].astype(str) == uid.strip()]

    st.write(f"**{len(view):,} users**")

    show = [
        "user_id", "Cluster", "Cluster_Name", "churn_risk_score",
        "engagement_score", "sessions_per_week", "days_since_last_login"
    ]

    st.dataframe(view[show].head(1000), use_container_width=True, hide_index=True)

    st.download_button(
        "Download this list (CSV)",
        view[show].to_csv(index=False).encode(),
        file_name="cluster_users.csv",
        mime="text/csv"
    )


# 6. BUSINESS ACTIONS
elif page == "Business Actions":

    st.title("Business insights & action mapping")

    rows = []

    for c in sorted(names):
        sub = dfc[dfc["Cluster"] == c]

        rows.append({
            "Cluster": label_for(c),
            "Users": len(sub),
            "Avg churn risk": round(sub["churn_risk_score"].mean(), 3),
            "Avg engagement": round(sub["engagement_score"].mean(), 2),
            "Recommended action": actions[c]
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Highest churn-risk users (top 20%)")

    threshold = dfc["churn_risk_score"].quantile(0.8)
    risky = dfc[dfc["churn_risk_score"] >= threshold]

    st.write(
        f"{len(risky):,} users with churn_risk_score >= {threshold:.2f} "
        "(flag for retention campaigns)"
    )

    st.dataframe(
        risky.sort_values("churn_risk_score", ascending=False)[
            ["user_id", "Cluster_Name", "churn_risk_score", "engagement_score"]
        ].head(500),
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download retention target list (CSV)",
        risky[["user_id", "Cluster_Name", "churn_risk_score"]].to_csv(index=False).encode(),
        file_name="retention_targets.csv",
        mime="text/csv"
    )

    st.caption(
        "Cluster names are assigned by ranking clusters on engagement level "
        "and feature interaction, so they stay correct even if cluster numbers change."
    )


# 7. PREDICT NEW USER
else:

    st.title("Predict a user's cluster")
    st.write("Enter behaviour values (defaults = dataset averages).")

    values = {}
    columns = st.columns(2)

    for i, feature in enumerate(FEATURES):
        minimum = float(df[feature].min())
        maximum = float(df[feature].max())
        average = float(df[feature].mean())

        values[feature] = columns[i % 2].number_input(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=average
        )

    if st.button("Predict cluster"):

        new_user = pd.DataFrame([values])[FEATURES]
        scaled_user = scaler.transform(new_user)

        cluster = int(model.predict(scaled_user)[0])

        st.success(f"Cluster {cluster}: **{names[cluster]}**")
        st.info(f"Recommended action: {actions[cluster]}")
