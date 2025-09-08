import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pycountry

# Set page configuration
st.set_page_config(
    page_title="World Happiness Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    # Replace with your actual file path
    dataset = pd.read_csv(r"Data\Cleand_Dataset_World_Happiness.csv")
    return dataset

dataset = load_data()

# Sidebar filters
st.sidebar.title("Filters")
years = sorted(dataset['Year'].unique())
selected_years = st.sidebar.multiselect(
    "Select Years",
    options=years,
    default=years
)

regions = sorted(dataset['Region'].unique())
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=regions,
    default=regions
)

# Filter data based on selections
filtered_data = dataset[
    (dataset['Year'].isin(selected_years)) & 
    (dataset['Region'].isin(selected_regions))
]

# Main content
st.title("🌍 World Happiness Dashboard")
st.markdown("Analyzing happiness scores across countries and regions from 2015 to 2019")

# Calculate proper country averages for the selected years
country_avg = filtered_data.groupby("Country")["Happiness Score"].mean().reset_index()

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    avg_score = filtered_data['Happiness Score'].mean()
    st.metric("Average Happiness Score", f"{avg_score:.2f}")
with col2:
    num_countries = filtered_data['Country'].nunique()
    st.metric("Number of Countries", num_countries)
with col3:
    # Get the country with the highest average score
    if not country_avg.empty:
        top_country_row = country_avg.loc[country_avg['Happiness Score'].idxmax()]
        top_country = top_country_row['Country']
        top_score = top_country_row['Happiness Score']
        st.metric("Happiest Country (Avg)", f"{top_country} ({top_score:.3f})")
    else:
        st.metric("Happiest Country (Avg)", "N/A")
with col4:
    # Get the country with the lowest average score
    if not country_avg.empty:
        bottom_country_row = country_avg.loc[country_avg['Happiness Score'].idxmin()]
        bottom_country = bottom_country_row['Country']
        bottom_score = bottom_country_row['Happiness Score']
        st.metric("Least Happy Country (Avg)", f"{bottom_country} ({bottom_score:.3f})")
    else:
        st.metric("Least Happy Country (Avg)", "N/A")

# Add a section to show the actual calculation for verification
with st.expander("ℹ️ Verification of Results"):
    st.write("**Top 5 Countries by Average Happiness Score (2015-2019):**")
    
    # Calculate the true average for all countries across all years
    all_countries_avg = dataset.groupby("Country")["Happiness Score"].mean().reset_index()
    top_5_all_time = all_countries_avg.nlargest(5, "Happiness Score")
    bottom_5_all_time = all_countries_avg.nsmallest(5, "Happiness Score")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Top 5 Countries:")
        for i, row in top_5_all_time.iterrows():
            st.write(f"{i+1}. {row['Country']}: {row['Happiness Score']:.3f}")
    
    with col2:
        st.write("Bottom 5 Countries:")
        for i, row in bottom_5_all_time.iterrows():
            st.write(f"{i+1}. {row['Country']}: {row['Happiness Score']:.3f}")

# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Regional Analysis", 
    "Trends Over Time", 
    "Country Rankings",
    "Correlation Analysis",
    "Clustering",
    "Feature Importance"
])

with tab1:
    st.header("Regional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Boxplot by region
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="Region", y="Happiness Score", data=filtered_data, palette="Set2")
        plt.xticks(rotation=45, ha="right")
        plt.title("Happiness Score Distribution by Region")
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    
    with col2:
        # Average happiness by region
        region_avg = filtered_data.groupby("Region")["Happiness Score"].mean().reset_index().sort_values(by="Happiness Score", ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Happiness Score", y="Region", data=region_avg, palette="viridis")
        plt.title("Average Happiness Score by Region")
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

with tab2:
    st.header("Trends Over Time")
    
    # Yearly average happiness
    yearly_avg = filtered_data.groupby("Year")["Happiness Score"].mean().reset_index()
    yearly_avg["Year"] = yearly_avg["Year"].astype(int)
    
    fig = px.line(yearly_avg, x="Year", y="Happiness Score", 
                  title="Average Happiness Score Over Time",
                  markers=True)
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional changes over time
    region_year_avg = filtered_data.groupby(["Region", "Year"])["Happiness Score"].mean().reset_index()
    fig = px.line(region_year_avg, x="Year", y="Happiness Score", color="Region",
                  title="Happiness Score Trends by Region",
                  markers=True)
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Country Rankings")
    
    # Sort by average happiness score
    country_avg_sorted = country_avg.sort_values(by="Happiness Score", ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Happiest Countries (Average)")
        top_10 = country_avg_sorted.head(10)
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Happiness Score", y="Country", data=top_10, palette="Blues_r")
        plt.title("Top 10 Happiest Countries (Average)")
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    
    with col2:
        st.subheader("Bottom 10 Least Happy Countries (Average)")
        bottom_10 = country_avg_sorted.tail(10)
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Happiness Score", y="Country", data=bottom_10, palette="Reds")
        plt.title("Bottom 10 Least Happy Countries (Average)")
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    
    # Interactive table
    st.subheader("Full Country Rankings (Average)")
    country_avg_sorted_renamed = country_avg_sorted.rename(columns={"Happiness Score": "Average Happiness Score"})
    country_avg_sorted_renamed["Rank"] = range(1, len(country_avg_sorted_renamed) + 1)
    st.dataframe(country_avg_sorted_renamed[["Rank", "Country", "Average Happiness Score"]], use_container_width=True)

with tab4:
    st.header("Correlation Analysis")
    
    # Correlation heatmap
    numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
    corr = filtered_data[numeric_cols].corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)
    plt.clf()
    
    # Scatter plot with selectable variables
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("X-axis variable", options=numeric_cols, index=0)
    with col2:
        y_var = st.selectbox("Y-axis variable", options=numeric_cols, index=1)
    
    fig = px.scatter(filtered_data, x=x_var, y=y_var, color="Region",
                     hover_name="Country", hover_data=["Year"],
                     title=f"{y_var} vs {x_var}")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Country Clustering")
    
    # Select features for clustering
    features = ["Happiness Score", "Economy", "Social Support", "Freedom", 
                "Generosity", "Trust On Govt", "Life Expectancy"]
    
    # Country-wise averages across years
    country_features = filtered_data.groupby("Country")[features].mean().reset_index()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(country_features[features])
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    country_features["Cluster"] = kmeans.fit_predict(X_scaled)
    
    # Map clusters to labels
    cluster_means = country_features.groupby("Cluster")["Happiness Score"].mean().sort_values()
    labels = {cluster_means.index[0]: "Low Happiness",
              cluster_means.index[1]: "Medium Happiness",
              cluster_means.index[2]: "High Happiness"}
    
    country_features["Happiness_Level"] = country_features["Cluster"].map(labels)
    
    # Scatter plot
    fig = px.scatter(country_features, x="Economy", y="Happiness Score",
                     color="Happiness_Level", hover_name="Country",
                     title="Country Clusters Based on Happiness Factors")
    st.plotly_chart(fig, use_container_width=True)
    
    # World map
    try:
        country_features["ISO_Code"] = country_features["Country"].map(
            lambda x: pycountry.countries.lookup(x).alpha_3 if x in [c.name for c in pycountry.countries] else None
        )
        
        fig = px.choropleth(
            country_features,
            locations="ISO_Code",
            color="Happiness_Level",
            hover_name="Country",
            hover_data={"Happiness Score": True, "ISO_Code": False},
            color_discrete_map={
                "Low Happiness": "red",
                "Medium Happiness": "orange",
                "High Happiness": "green"
            },
            title="Global Happiness Clusters"
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Could not generate world map due to country name issues")

with tab6:
    st.header("Feature Importance")
    
    # Features and target
    features = ["Economy", "Social Support", "Freedom", 
                "Generosity", "Trust On Govt", "Life Expectancy", "Dystopia Residual"]
    X = filtered_data[features]
    y = filtered_data["Happiness Score"]
    
    # Train a Random Forest
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=importance, palette="viridis")
    plt.title("Feature Importance for Happiness Score")
    st.pyplot(plt)
    plt.clf()
    
    st.dataframe(importance, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Data source: World Happiness Report (2015-2019)")