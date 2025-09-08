# What-Makes-a-Country-Happy

# 🌍 What Makes a Country Happy?  
*An Exploratory Data Science Project on Global Happiness (2015–2019)*  

## 📌 Overview  
This project analyzes **World Happiness Report data (2015–2019)** to identify the key drivers of happiness, study regional and year-wise trends, and apply clustering techniques to group countries by happiness profiles.  

The goal is to understand:  
- *What factors make people happier across nations?*  
- *How do different regions compare in terms of well-being?*  
- *Can clustering reveal distinct “happiness profiles” of countries?*  

---

## 🔑 Key Findings  

### 1️⃣ Drivers of Happiness  
- **Top 3 drivers**: 🏦 Economy, ❤️ Life Expectancy, 👨‍👩‍👧 Social Support  
- **Secondary contributors**: Freedom, Trust in Government  
- **Least impactful globally**: Generosity, Year  

---

### 2️⃣ Year-wise Trends  
- 🌎 **Global happiness stable** (2015–2018, avg ≈ 5.4), with a mild upward trend.  
- 😀 **Happiest Countries**: Nordic nations, led by Finland (2017–2019).  
- 😔 **Least Happy Countries**: Conflict-affected regions (Burundi, Afghanistan).  
- 📉 Persistent **4.5-point happiness gap** between top and bottom nations.  

| Year | Top Country   | Top Score | Bottom Country | Bottom Score |
|------|---------------|-----------|----------------|--------------|
| 2015 | Switzerland   | 7.587     | Togo           | 2.839        |
| 2016 | Denmark       | 7.526     | Burundi        | 2.905        |
| 2017 | Norway        | 7.537     | Burundi        | 2.905        |
| 2018 | Finland       | 7.632     | Burundi        | 2.905        |
| 2019 | Finland       | 7.769     | Afghanistan    | 3.203        |

---

### 3️⃣ Regional Insights  
- 🌏 **Happiest Regions**: Australia & New Zealand, North America, Western Europe (avg > 7).  
- 🌎 **Lowest Scores**: Sub-Saharan Africa, South Asia, Middle East (avg < 5).  
- 📈 **Most Improved**: Sub-Saharan Africa (+0.26).  
- 📉 **Most Declined**: Latin America & Caribbean (–0.19).  

---

### 4️⃣ Dystopia Residual  
- Happiest countries have **higher dystopia residuals** (≈ 2.49) vs. others (≈ 1.96).  
- ❌ So being “happier” doesn’t mean less dystopia residual.  

---

### 5️⃣ Country Averages (2015–2019)  
- 📈 **Happiest**: Denmark (7.546)  
- 📉 **Least Happy**: Burundi (3.079)  

---

### 6️⃣ Feature Importance (via ML model)  
| Feature            | Importance |
|--------------------|------------|
| Economy            | **0.435** |
| Life Expectancy    | **0.242** |
| Dystopia Residual  | 0.169      |
| Social Support     | 0.056      |
| Freedom            | 0.045      |
| Trust in Govt      | 0.031      |
| Generosity         | 0.023      |

➡️ **Economy + Life Expectancy = ~68% of happiness explained.**

---

## 📊 Methods Used  
- **EDA**: Heatmaps, scatterplots, regional trends  
- **Clustering**: K-Means to group countries into *Low, Mid, High Happiness Profiles*  
- **Feature Importance**: Machine learning to rank drivers of happiness  
- **Visualization**: Choropleth maps, year-wise comparisons  

---

## 🛠 Tech Stack  
- Python (Pandas, NumPy, Scikit-learn, Seaborn, Plotly)  
- Data: [World Happiness Report (2015–2019)](https://worldhappiness.report/)  

---

## 🚀 How to Run  
```bash
git clone https://github.com/Atanu-Manna2003/What-Makes-a-Country-Happy.git
cd What-Makes-a-Country-Happy
pip install -r requirements.txt
python app.py
