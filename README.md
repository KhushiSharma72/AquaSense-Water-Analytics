# 💧 AquaSense – Water Consumption Prediction & Analytics Platform

**A data analytics and machine learning project using PySpark, Python, and Streamlit to explore water consumption patterns and generate consumption predictions.**

🌐 **Live Dashboard:** [Open AquaSense](https://aquasense-water-analytics-6kdqyeq35eymjhvlxklld7.streamlit.app/)

📂 **GitHub Repository:** [AquaSense – Water Analytics](https://github.com/KhushiSharma72/AquaSense-Water-Analytics)

---

## 📌 Project Overview

AquaSense is an interactive water consumption analytics and prediction platform built using NYC water consumption data from 2013 to May 2025.

The project focuses on analyzing historical water consumption, identifying trends across boroughs and developments, and exploring machine learning-based consumption predictions.

It combines big data processing, exploratory data analysis, machine learning, and interactive visualization in one platform.

## 🎯 Project Objectives

- Analyze historical water consumption patterns.
- Identify consumption trends across different boroughs.
- Explore monthly and yearly consumption variations.
- Understand consumption patterns across development and rate classes.
- Build a machine learning model to estimate water consumption.
- Present insights through an interactive dashboard.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| PySpark | Large-scale data processing |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive charts |
| Streamlit | Interactive web dashboard |
| Spark MLlib | Machine learning pipeline |
| Git & GitHub | Version control and hosting |

## 📊 Dataset

**Dataset:** Water Consumption and Cost (2013 – May 2025)

**Source:** [NYC Open Data](https://data.cityofnewyork.us/Housing-Development/Water-Consumption-And-Cost-2013-May-2025/66be-66yr)

The original dataset contains approximately 66,432 records and 25 columns.

Important features include:

- Development Name
- Borough
- Rate Class
- Revenue Month
- Meter Number
- Estimated
- Consumption (HCF)

**Data preparation:**
- Removed exact duplicate records.
- Filtered records with meter numbers ending in `-Sewer`.
- Converted consumption values into numeric format.
- Removed records with missing consumption values.
- Prepared the cleaned dataset for analysis and modeling.

After preprocessing, **56,553 records** were retained.

## 🔍 Exploratory Data Analysis

The project explores:

1. Consumption distribution.
2. Median consumption across boroughs.
3. Monthly water consumption trends.
4. Consumption differences across developments.

Median values are used in relevant comparisons to reduce the influence of extreme consumption values.

## 🤖 Machine Learning

A Random Forest Regressor was implemented using Spark MLlib.

**Model workflow:**

1. Feature selection.
2. Categorical feature indexing.
3. One-hot encoding.
4. Feature vector assembly.
5. Time-based train-test splitting.
6. Model training and evaluation.
7. Prediction generation.

The target variable was log-transformed during training to help manage highly skewed consumption values.

### Model Evaluation

The dataset was divided using a time-based split:

- Training records: 55,278
- Testing records: 1,275

| Evaluation Metric | Result |
|---|---:|
| MAE | 135.70 |
| RMSE | 398.52 |
| R² Score | 0.1417 |

**Interpretation:** The model achieved an R² score of approximately 0.14 on the test data. This indicates limited explanatory performance, so the predictions should be treated as exploratory estimates rather than highly accurate forecasts.

## 📈 Interactive Dashboard

AquaSense provides four dashboard sections:

- **Overview:** Key metrics and consumption insights.
- **Trend Explorer:** Monthly trends, borough comparisons, and consumption patterns.
- **Model Insights:** Actual vs. predicted values and prediction error analysis.
- **Data Explorer:** Filter, search, explore, and download records.

### Dashboard Features

- Interactive filters.
- Borough and year selection.
- Dynamic visualizations.
- Consumption analysis.
- Model evaluation charts.
- Downloadable CSV data.

## 🏗️ Project Structure

```text
AquaSense/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│       └── water_cleaned.csv
│
├── reports/
│   └── predictions.csv
│
├── src/
│   ├── clean_data.py
│   ├── eda.py
│   ├── prepare_ml.py
│   └── train_model.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/KhushiSharma72/AquaSense-Water-Analytics.git
```

### 2. Navigate to the project directory

```bash
cd AquaSense-Water-Analytics
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open locally in your browser.

**Note:** The dashboard uses the processed dataset and prediction CSV included in the repository. Running the full PySpark processing and training pipeline separately requires the original dataset and a compatible Java/PySpark environment.

## 🚀 Deployment

The AquaSense dashboard is deployed using **Streamlit Community Cloud**.

🔗 [Visit the Live Application](https://aquasense-water-analytics-6kdqyeq35eymjhvlxklld7.streamlit.app/)

## 🔮 Future Improvements

- Improve model performance through additional feature engineering.
- Explore more machine learning algorithms.
- Include additional factors affecting water consumption.
- Develop more robust forecasting methods.
- Add geographical visualizations and deeper analytics.

## 👩‍💻 Author

**Khushi Sharma**

B.Tech Computer Science and Engineering  
Lovely Professional University

GitHub: [KhushiSharma72](https://github.com/KhushiSharma72)

Portfolio: [Visit My Portfolio](https://khushisharma1.netlify.app/)

---

⭐ If you find this project interesting, feel free to explore the repository and live dashboard!
