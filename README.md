# App User Behavior Segmentation Using Machine Learning

## Project Overview
This project focuses on segmenting **50,000 app users** based on their usage patterns and engagement levels using Unsupervised Machine Learning.

The goal is to understand user behavior, identify different user groups, and provide insights that can help improve user engagement and retention.

## Objectives
- Analyze app user behavior and engagement patterns.
- Identify user groups using clustering algorithms.
- Visualize clusters and understand their characteristics.
- Provide insights to support user retention strategies.

## Technologies Used
- **Programming:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Machine Learning:** K-Means, DBSCAN, Hierarchical Clustering
- **Visualization:** PCA, Streamlit

## Project Workflow
1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature selection and standardization
4. Clustering using K-Means
5. Cluster evaluation using the Elbow Method and Silhouette Score
6. PCA for dimensionality reduction and visualization
7. User segmentation and business insights
8. Interactive Streamlit dashboard

## User Segments
The project report identifies four user groups:

| Cluster | User Segment |
|---|---|
| 0 | High Engagement Users |
| 1 | Moderate Engagement Users |
| 2 | Low Engagement / At-Risk Users |
| 3 | Occasional Users |

*Note: Cluster labels are based on the project report. Cluster IDs should be verified against the final trained model.*

## Dashboard Features
- User segmentation overview
- Interactive filters
- Engagement analysis
- Cluster comparison
- User-level details
- Downloadable filtered data

## How to Run the Project

**1. Clone the repository**
```bash
git clone https://github.com/lokesh007-crypto/REPOSITORY-NAME.git
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit dashboard**
```bash
streamlit run app.py
```

## Author
**Lokesh P**

GitHub: [lokesh007-crypto](https://github.com/lokesh007-crypto)

---

*This project demonstrates the application of Unsupervised Machine Learning for understanding user behavior and generating actionable business insights.*
