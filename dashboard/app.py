
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AquaSense | Water Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# THEME AND CUSTOM CSS
# =====================================================

NAVY = "#0B1220"
PANEL = "#111D30"
CYAN = "#22D3EE"
BLUE = "#3B82F6"
PURPLE = "#A78BFA"
TEXT = "#E5EDF7"
MUTED = "#94A3B8"

st.markdown("""
<style>
    .stApp {
        background-color: #0B1220;
        color: #E5EDF7;
    }

    [data-testid="stSidebar"] {
        background-color: #101B2D;
        border-right: 1px solid #26364D;
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(115deg, #14243B, #102D42);
        border: 1px solid #25445D;
        border-radius: 18px;
        padding: 25px 28px;
        margin-bottom: 22px;
    }

    .hero-title {
        color: #F1F7FF;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        color: #A9C0D5;
        font-size: 14px;
        margin-top: 8px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #14243A, #101D30);
        border: 1px solid #263F59;
        padding: 17px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #A9C0D5;
    }

    div[data-testid="stMetricValue"] {
        color: #F1F7FF;
        font-size: 25px;
        font-weight: 700;
    }

    .insight-card {
        background: #14243A;
        border: 1px solid #28445E;
        border-left: 4px solid #22D3EE;
        border-radius: 10px;
        padding: 15px;
        min-height: 105px;
        margin-bottom: 10px;
    }

    .insight-label {
        color: #94A3B8;
        font-size: 12px;
        margin-bottom: 7px;
    }

    .insight-value {
        color: #E5EDF7;
        font-size: 18px;
        font-weight: 700;
    }

    .section-note {
        color: #94A3B8;
        font-size: 13px;
    }

    h1, h2, h3 {
        color: #F1F7FF !important;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .stDownloadButton button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    clean = pd.read_csv(
        BASE_DIR / "data" / "processed" / "water_cleaned.csv"
    )

    predictions = pd.read_csv(
        BASE_DIR / "reports" / "predictions.csv"
    )

    for data in [clean, predictions]:

        data["Revenue Month"] = pd.to_datetime(
            data["Revenue Month"], errors="coerce"
        )

        data["Consumption"] = pd.to_numeric(
            data["Consumption"], errors="coerce"
        )

        data["Year"] = data["Revenue Month"].dt.year
        data["Month"] = data["Revenue Month"].dt.month

    predictions["Prediction"] = pd.to_numeric(
        predictions["Prediction"], errors="coerce"
    )

    clean = clean.dropna(
        subset=["Revenue Month", "Consumption"]
    )

    predictions = predictions.dropna(
        subset=["Revenue Month", "Consumption", "Prediction"]
    )

    return clean, predictions


try:
    df, pred = load_data()

except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.stop()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def format_hcf(value):

    if pd.isna(value):
        return "N/A"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M HCF"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K HCF"

    return f"{value:,.2f} HCF"


def chart_layout(fig, height=390):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        height=height,
        margin=dict(l=15, r=15, t=55, b=15),
        legend_title_text=""
    )

    fig.update_xaxes(
        gridcolor="#26364D",
        zerolinecolor="#26364D"
    )

    fig.update_yaxes(
        gridcolor="#26364D",
        zerolinecolor="#26364D"
    )

    return fig


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.markdown("## 💧 AquaSense")
st.sidebar.caption("WATER CONSUMPTION ANALYTICS")
st.sidebar.divider()

borough_list = sorted(
    df["Borough"].dropna().astype(str).unique()
)

selected_borough = st.sidebar.selectbox(
    "Select Borough",
    ["All Boroughs"] + borough_list
)

available_years = sorted(
    df["Year"].dropna().astype(int).unique()
)

if len(available_years) == 0:
    st.error("No valid year information found.")
    st.stop()

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(min(available_years)),
    max_value=int(max(available_years)),
    value=(
        int(min(available_years)),
        int(max(available_years))
    )
)

st.sidebar.divider()

st.sidebar.markdown("### Display Options")

measurement = st.sidebar.selectbox(
    "Consumption Measurement",
    ["Median", "Average", "Total"]
)

show_estimated = st.sidebar.checkbox(
    "Include estimated records",
    value=True
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")
st.sidebar.caption("Dataset: NYC Water Consumption")
st.sidebar.caption("Model: Random Forest Regression")
st.sidebar.caption("Technology: PySpark + MLlib + Streamlit")


# =====================================================
# FILTER DATA
# =====================================================

filtered_df = df[
    df["Year"].between(year_range[0], year_range[1])
].copy()

filtered_pred = pred[
    pred["Year"].between(year_range[0], year_range[1])
].copy()

if selected_borough != "All Boroughs":

    filtered_df = filtered_df[
        filtered_df["Borough"].astype(str) == selected_borough
    ]

    filtered_pred = filtered_pred[
        filtered_pred["Borough"].astype(str) == selected_borough
    ]

if not show_estimated and "Estimated" in filtered_df.columns:

    estimated_values = (
        filtered_df["Estimated"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    filtered_df = filtered_df[
        estimated_values.isin(["false", "0", "no", "n"])
    ]


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">💧 AquaSense</div>
    <div class="hero-subtitle">
        Water Consumption Prediction & Analytics Platform
        <br>
        Explore historical consumption, discover patterns
        and evaluate machine learning predictions.
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# KPI SECTION
# =====================================================

st.subheader("Dashboard Overview")

total_records = len(filtered_df)

total_consumption = filtered_df["Consumption"].sum()

median_consumption = filtered_df["Consumption"].median()

average_consumption = filtered_df["Consumption"].mean()

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Records",
    f"{total_records:,}"
)

k2.metric(
    "Total Consumption",
    format_hcf(total_consumption)
)

k3.metric(
    "Median Consumption",
    format_hcf(median_consumption)
)

k4.metric(
    "Average Consumption",
    format_hcf(average_consumption)
)

st.caption(
    f"Showing {total_records:,} records | "
    f"Years: {year_range[0]}–{year_range[1]} | "
    f"Borough: {selected_borough}"
)

st.divider()


# =====================================================
# AUTOMATIC INSIGHTS
# =====================================================

st.subheader("✨ Smart Data Insights")

if len(filtered_df) > 0:

    borough_summary = (
        filtered_df.groupby("Borough")["Consumption"]
        .median()
        .sort_values(ascending=False)
    )

    development_summary = (
        filtered_df.groupby("Development Name")["Consumption"]
        .median()
        .sort_values(ascending=False)
    )

    highest_borough = borough_summary.index[0]
    highest_borough_value = borough_summary.iloc[0]

    highest_development = development_summary.index[0]
    highest_development_value = development_summary.iloc[0]

    insight1, insight2, insight3 = st.columns(3)

    with insight1:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-label">Highest Median Borough</div>
            <div class="insight-value">{highest_borough}</div>
            <div class="section-note">
                {format_hcf(highest_borough_value)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with insight2:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-label">Highest Median Development</div>
            <div class="insight-value">{str(highest_development)[:32]}</div>
            <div class="section-note">
                {format_hcf(highest_development_value)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with insight3:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-label">Unique Developments</div>
            <div class="insight-value">
                {filtered_df["Development Name"].nunique():,}
            </div>
            <div class="section-note">
                In the selected records
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("No records match the selected filters.")


st.divider()


# =====================================================
# MAIN TABS
# =====================================================

overview_tab, trends_tab, model_tab, explorer_tab = st.tabs([
    "📊 Overview",
    "📈 Trend Explorer",
    "🤖 Model Insights",
    "📋 Data Explorer"
])


# =====================================================
# TAB 1: OVERVIEW
# =====================================================

with overview_tab:

    st.subheader("Consumption by Borough")

    if len(filtered_df) > 0:

        aggregation = {
            "Median": "median",
            "Average": "mean",
            "Total": "sum"
        }[measurement]

        borough_data = (
            filtered_df.groupby("Borough")["Consumption"]
            .agg(aggregation)
            .reset_index()
            .sort_values("Consumption", ascending=False)
        )

        fig = px.bar(
            borough_data,
            x="Borough",
            y="Consumption",
            color="Borough",
            title=f"{measurement} Consumption by Borough",
            color_discrete_sequence=[
                CYAN, BLUE, PURPLE, "#34D399",
                "#FBBF24", "#F472B6", "#818CF8"
            ]
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            chart_layout(fig),
            width="stretch"
        )

        left, right = st.columns(2)

        with left:

            top_developments = (
                filtered_df.groupby("Development Name")["Consumption"]
                .median()
                .nlargest(10)
                .sort_values()
                .reset_index()
            )

            fig = px.bar(
                top_developments,
                x="Consumption",
                y="Development Name",
                orientation="h",
                title="Top 10 Developments by Median Consumption",
                color="Consumption",
                color_continuous_scale="Teal"
            )

            st.plotly_chart(
                chart_layout(fig),
                width="stretch"
            )

        with right:

            fig = px.histogram(
                filtered_df,
                x="Consumption",
                nbins=45,
                title="Consumption Distribution",
                color_discrete_sequence=[CYAN]
            )

            st.plotly_chart(
                chart_layout(fig),
                width="stretch"
            )

    else:
        st.info("No data available for these filters.")


# =====================================================
# TAB 2: TREND EXPLORER
# =====================================================

with trends_tab:

    st.subheader("Explore Consumption Trends")

    trend_metric = st.radio(
        "Select measurement",
        ["Median", "Average", "Total"],
        horizontal=True,
        key="trend_metric"
    )

    trend_aggregation = {
        "Median": "median",
        "Average": "mean",
        "Total": "sum"
    }[trend_metric]

    if len(filtered_df) > 0:

        monthly = (
            filtered_df.groupby("Revenue Month")["Consumption"]
            .agg(trend_aggregation)
            .reset_index()
            .sort_values("Revenue Month")
        )

        fig = px.line(
            monthly,
            x="Revenue Month",
            y="Consumption",
            markers=True,
            title=f"Monthly {trend_metric} Consumption",
            color_discrete_sequence=[CYAN]
        )

        fig.update_traces(line=dict(width=3))

        st.plotly_chart(
            chart_layout(fig, 420),
            width="stretch"
        )

        st.subheader("Year vs Month Heatmap")

        heatmap_data = filtered_df.copy()

        heatmap_data["Month Name"] = (
            heatmap_data["Revenue Month"].dt.month_name()
        )

        month_order = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]

        heatmap = heatmap_data.pivot_table(
            index="Year",
            columns="Month",
            values="Consumption",
            aggfunc="median"
        )

        heatmap = heatmap.reindex(columns=range(1, 13))
        heatmap.columns = [
            pd.Timestamp(2000, month, 1).strftime("%b")
            for month in heatmap.columns
        ]

        fig = px.imshow(
            heatmap,
            labels={
                "x": "Month",
                "y": "Year",
                "color": "Median HCF"
            },
            title="Monthly Consumption Heatmap",
            color_continuous_scale="Teal",
            aspect="auto"
        )

        st.plotly_chart(
            chart_layout(fig, 430),
            width="stretch"
        )

        st.caption(
            "The heatmap shows median consumption for each year and month. "
            "Blank cells indicate months without available records."
        )

    else:
        st.info("No trend data available.")


# =====================================================
# TAB 3: MODEL INSIGHTS
# =====================================================

with model_tab:

    st.subheader("Random Forest Model Performance")

    st.caption(
        "Model trained on records before January 2025 "
        "and tested on records from January 2025 onward."
    )

    if len(filtered_pred) > 0:

        actual = filtered_pred["Consumption"].to_numpy()
        predicted = filtered_pred["Prediction"].to_numpy()

        errors = actual - predicted
        absolute_errors = np.abs(errors)

        mae = np.mean(absolute_errors)
        rmse = np.sqrt(np.mean(errors ** 2))

        denominator = np.sum((actual - np.mean(actual)) ** 2)

        r2 = (
            1 - np.sum(errors ** 2) / denominator
            if denominator > 0 else np.nan
        )

        a, b, c = st.columns(3)

        a.metric("Mean Absolute Error", f"{mae:.2f} HCF")
        b.metric("Root Mean Squared Error", f"{rmse:.2f} HCF")
        c.metric(
            "R² Score",
            f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        )

        st.divider()

        st.subheader("Actual vs Predicted")

        fig = px.scatter(
            filtered_pred,
            x="Consumption",
            y="Prediction",
            color="Borough",
            hover_data=[
                "Development Name",
                "Revenue Month"
            ],
            title="Actual Consumption vs Predicted Consumption",
            opacity=0.7
        )

        maximum = max(
            float(np.max(actual)),
            float(np.max(predicted))
        )

        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=maximum,
            y1=maximum,
            line=dict(
                color="#FBBF24",
                dash="dash",
                width=2
            )
        )

        st.plotly_chart(
            chart_layout(fig, 430),
            width="stretch"
        )

        st.caption(
            "The dashed line represents perfect predictions. "
            "Points closer to this line have smaller prediction errors."
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("Prediction Error Distribution")

            error_df = pd.DataFrame({
                "Prediction Error": errors
            })

            fig = px.histogram(
                error_df,
                x="Prediction Error",
                nbins=40,
                title="Actual minus Predicted",
                color_discrete_sequence=[PURPLE]
            )

            st.plotly_chart(
                chart_layout(fig, 350),
                width="stretch"
            )

        with right:

            st.subheader("Absolute Error Distribution")

            error_df["Absolute Error"] = absolute_errors

            fig = px.box(
                error_df,
                y="Absolute Error",
                title="Spread of Absolute Prediction Errors",
                color_discrete_sequence=[CYAN]
            )

            st.plotly_chart(
                chart_layout(fig, 350),
                width="stretch"
            )

        st.divider()

        st.subheader("Prediction Comparison Table")

        comparison = filtered_pred[
            [
                "Revenue Month",
                "Development Name",
                "Borough",
                "Consumption",
                "Prediction"
            ]
        ].copy()

        comparison["Absolute Error"] = (
            comparison["Consumption"] - comparison["Prediction"]
        ).abs()

        st.dataframe(
            comparison.sort_values(
                "Absolute Error",
                ascending=False
            ),
            width="stretch",
            hide_index=True
        )

    else:
        st.info(
            "No test predictions are available for the selected filters."
        )


# =====================================================
# TAB 4: DATA EXPLORER
# =====================================================

with explorer_tab:

    st.subheader("Explore the Dataset")

    search = st.text_input(
        "🔎 Search Development",
        placeholder="Type a development name..."
    )

    explorer_df = filtered_df.copy()

    if search:

        explorer_df = explorer_df[
            explorer_df["Development Name"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.write(f"**Matching records:** {len(explorer_df):,}")

    display_columns = [
        column for column in [
            "Development Name",
            "Borough",
            "Rate Class",
            "Revenue Month",
            "Consumption",
            "Estimated"
        ]
        if column in explorer_df.columns
    ]

    st.dataframe(
        explorer_df[display_columns],
        width="stretch",
        hide_index=True
    )

    csv_data = explorer_df[display_columns].to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download Filtered Records",
        data=csv_data,
        file_name="water_consumption_filtered.csv",
        mime="text/csv"
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94A3B8; padding:10px;">
        💧 <b>AquaSense</b> | Water Consumption Prediction Project
        <br>
        Built with Python • PySpark • MLlib • Streamlit • Plotly
    </div>
    """,
    unsafe_allow_html=True
)
