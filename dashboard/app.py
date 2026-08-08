import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG + NETFLIX THEME
# =============================================================================
st.set_page_config(page_title="Netflix Churn Analytics", layout="wide", page_icon="🎬")

NETFLIX_RED = "#E50914"
BG_BLACK = "#141414"
CARD_BG = "#1F1F1F"
TEXT_GRAY = "#B3B3B3"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_BLACK}; color: white; }}
    section[data-testid="stSidebar"] {{ background-color: {CARD_BG}; }}
    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
    }}
    div[data-testid="stMetricValue"] {{ color: white; }}
    h1, h2, h3 {{ color: white; }}
    .netflix-title {{
        color: {NETFLIX_RED};
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 0px;
    }}
    .tagline {{
        color: {TEXT_GRAY};
        font-style: italic;
        font-size: 15px;
    }}
</style>
""", unsafe_allow_html=True)

# Plotly template matching the dashboard theme -- reused across every chart
# so all visuals look consistent without repeating layout config each time.
PLOTLY_TEMPLATE = go.layout.Template()
PLOTLY_TEMPLATE.layout = go.Layout(
    paper_bgcolor=BG_BLACK,
    plot_bgcolor=BG_BLACK,
    font=dict(color="white"),
    colorway=[NETFLIX_RED, "#8B0000", "#B3B3B3", "#5A5A5A", "#FF4D4D"],
    xaxis=dict(gridcolor="#333", zerolinecolor="#333"),
    yaxis=dict(gridcolor="#333", zerolinecolor="#333"),
)


# =============================================================================
# DATA LOADING -- cached so re-navigating pages doesn't re-read CSVs each time
# =============================================================================
@st.cache_data
def load_data():
    """
    Loads all dashboard CSVs from data/processed/. Cached with st.cache_data
    so Streamlit only reads from disk once per session, not on every
    page interaction -- important since some of these files (genre/country
    distribution) are large due to the explode step from Phase 2.
    """
    data = {}
    base = "data/processed/"
    data["catalog"] = pd.read_csv(base + "dashboard_catalog_overview.csv")
    data["genre"] = pd.read_csv(base + "dashboard_genre_distribution.csv")
    data["country"] = pd.read_csv(base + "dashboard_country_distribution.csv")
    data["segments"] = pd.read_csv(base + "dashboard_segments.csv")
    data["churn_risk"] = pd.read_csv(base + "dashboard_churn_risk.csv")
    data["shap"] = pd.read_csv(base + "dashboard_shap_importance.csv")
    data["recommendations"] = pd.read_csv(base + "dashboard_recommendations.csv")
    return data

try:
    data = load_data()
except FileNotFoundError as e:
    st.error(f"Could not find a required data file: {e}. Make sure you're running "
             f"this from the project root, and that all dashboard_*.csv exports exist "
             f"in data/processed/.")
    st.stop()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.markdown(f'<div class="netflix-title">NETFLIX</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="tagline">Churn Analytics</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Content Overview", "Viewer Segments", "Churn Risk", "Recommendations"],
    label_visibility="collapsed",
)


# =============================================================================
# PAGE: HOME
# =============================================================================
if page == "Home":
    st.markdown('<div class="netflix-title">NETFLIX</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">"Every subscriber has a story. Some just disappear into the shadows."</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Content", f"{len(data['catalog']):,}")
    col2.metric("Users Analyzed", f"{len(data['segments']):,}")
    churn_rate = data["segments"]["churned_next_30d"].mean()
    col3.metric("Churn Rate", f"{churn_rate:.1%}")
    at_risk = (data["churn_risk"]["churn_probability"] > 0.5).sum()
    col4.metric("Users at Risk", f"{at_risk}")

    st.write("")
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.subheader("Segment Composition")

        seg_counts = (
            data["segments"]["segment"]
            .value_counts()
            .reset_index()
        )
        seg_counts.columns = ["segment", "count"]

        fig = px.pie(
            seg_counts,
            names="segment",
            values="count",
            hole=0.55,
            template=PLOTLY_TEMPLATE
        )

        fig.update_traces(
            textinfo="percent",
            textposition="inside"
        )

        fig.update_layout(
            height=350,
            margin=dict(t=10, b=10, l=10, r=10),

            # Put legend outside donut
            legend=dict(
                orientation="v",
                x=1.02,
                y=0.5,
                xanchor="left",
                yanchor="middle"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.subheader("Top Churn Drivers")
        top_shap = data["shap"].sort_values("mean_abs_shap", ascending=True).tail(5)
        fig = px.bar(top_shap, x="mean_abs_shap", y="feature", orientation="h",
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(xaxis_title="Impact on Prediction", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.info("**Quick Insight:** Completion rate is the strongest single predictor "
                "of churn risk across all segments.")


# =============================================================================
# PAGE: CONTENT OVERVIEW
# =============================================================================
elif page == "Content Overview":
    st.header("Content Overview")

    col1, col2 = st.columns(2)
    with col1:
        genre_counts = data["genre"]["genre_list"].value_counts().head(15).reset_index()
        genre_counts.columns = ["genre", "count"]
        fig = px.bar(genre_counts, x="count", y="genre", orientation="h",
                     template=PLOTLY_TEMPLATE, title="Content by Genre")
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                           xaxis_title="Number of Titles", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rating_counts = data["catalog"]["rating"].value_counts().reset_index()
        rating_counts.columns = ["rating", "count"]
        fig = px.bar(rating_counts, x="rating", y="count", template=PLOTLY_TEMPLATE,
                     title="Content by Rating")
        fig.update_layout(xaxis_title="Rating", yaxis_title="Number of Titles")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        country_counts = data["country"]["country_list"].value_counts().head(10).reset_index()
        country_counts.columns = ["country", "count"]
        fig = px.bar(country_counts, x="count", y="country", orientation="h",
                     template=PLOTLY_TEMPLATE, title="Content by Country")
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                           xaxis_title="Number of Titles", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        catalog = data["catalog"].copy()
        catalog["date_added"] = pd.to_datetime(catalog["date_added"], errors="coerce")
        catalog["year_added"] = catalog["date_added"].dt.year
        trend = catalog.groupby(["year_added", "type"]).size().reset_index(name="count")
        fig = px.line(trend, x="year_added", y="count", color="type",
                      template=PLOTLY_TEMPLATE, title="Content Added Over Time", markers=True)
        fig.update_layout(xaxis_title="Year", yaxis_title="Number of Titles")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: VIEWER SEGMENTS
# =============================================================================
elif page == "Viewer Segments":
    st.header("Viewer Segments")

    seg_profile = data["segments"].groupby("segment").agg(
        avg_completion_rate=("avg_completion_rate", "mean"),
        obs_avg_sessions_per_week=("obs_avg_sessions_per_week", "mean"),
        tenure_days=("tenure_days", "mean"),
        genre_diversity=("genre_diversity", "mean"),
        size=("user_id", "count"),
    ).round(2).reset_index()

    fig = px.scatter(seg_profile, x="avg_completion_rate", y="obs_avg_sessions_per_week",
                      color="segment", size="size", template=PLOTLY_TEMPLATE,
                      title="Engagement vs Completion by Segment", size_max=40)
    fig.update_layout(xaxis_title="Avg Completion Rate", yaxis_title="Avg Sessions per Week")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1, 1.5])
    with col1:
        seg_counts = data["segments"]["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.pie(seg_counts, names="segment", values="count", hole=0.5,
                     template=PLOTLY_TEMPLATE, title="Segment Sizes")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Segment Profiles")
        st.dataframe(seg_profile, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: CHURN RISK
# =============================================================================
elif page == "Churn Risk":
    st.header("Churn Risk")

    col1, col2 = st.columns(2)
    with col1:
        top_shap = data["shap"].sort_values("mean_abs_shap", ascending=True)
        fig = px.bar(top_shap, x="mean_abs_shap", y="feature", orientation="h",
                     template=PLOTLY_TEMPLATE, title="Top Churn Drivers")
        fig.update_layout(xaxis_title="Impact on Prediction", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_rate = data["segments"]["churned_next_30d"].mean()
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_rate * 100,
            number={"suffix": "%"},
            title={"text": "Overall Churn Rate"},
            gauge={
                "axis": {"range": [0, 15]},
                "bar": {"color": NETFLIX_RED},
                "threshold": {"line": {"color": "white", "width": 3}, "value": 3},
            },
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Avg Risk Score by Segment")
    risk_by_segment = data["churn_risk"].groupby("segment")["churn_probability"].mean().reset_index()
    fig = px.bar(risk_by_segment, x="segment", y="churn_probability", template=PLOTLY_TEMPLATE)
    fig.update_layout(yaxis_title="Avg Churn Probability", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest Risk Users")
    high_risk = data["churn_risk"].sort_values("churn_probability", ascending=False).head(20)
    st.dataframe(
        high_risk[["user_id", "segment", "churn_probability", "obs_avg_sessions_per_week", "avg_completion_rate"]],
        use_container_width=True, hide_index=True,
    )


# =============================================================================
# PAGE: RECOMMENDATIONS
# =============================================================================
elif page == "Recommendations":
    st.header("Recommendations")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Filter by Segment")
        segments_available = sorted(data["recommendations"]["segment"].unique())
        selected_segment = st.radio("Segment", ["All"] + segments_available, label_visibility="collapsed")

        st.write("")
        st.metric("Total Recommendations", len(data["recommendations"]))
        st.metric("Segments Covered", data["recommendations"]["segment"].nunique())

    with col2:
        method_counts = data["recommendations"]["method"].value_counts().reset_index()
        method_counts.columns = ["method", "count"]
        fig = px.pie(method_counts, names="method", values="count", hole=0.55,
                     template=PLOTLY_TEMPLATE, title="Recommendation Method Split")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recommended Titles by Segment")
    recs = data["recommendations"]
    if selected_segment != "All":
        recs = recs[recs["segment"] == selected_segment]
    st.dataframe(
        recs.sort_values(["segment", "score"], ascending=[True, False]),
        use_container_width=True, hide_index=True,
    )