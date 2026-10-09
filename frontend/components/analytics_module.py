import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import STOPWORDS, WordCloud

st.set_page_config(page_title="Analytics Module", page_icon="📊", layout="wide")

# ---------------------------------------------------------------- data
Basepath = Path(__file__).parent.parent / "datasets"
datapath = Basepath / "kolkata_cleaned_v5.csv"


@st.cache_data(show_spinner="Loading data...")
def load_data():
    return pd.read_csv(datapath)


new_df = load_data()

# --- Grouped data for map ---
grouped1 = (
    new_df.groupby("LOCALITY").agg({
        "PRICE_Cr": "mean",
        "PRICE_PER_SQFT(INR)": "mean",
        "AREA_SQFT": "mean",
        "LATITUDE": "mean",
        "LONGITUDE": "mean",
        "CITY_NUM": "first",
        "CITY_LABELS": "first",
    })
).reset_index()

# Build hover labels using LOCALITY column
grouped1["MAP_LABEL"] = grouped1.apply(
    lambda row: f"<b>{row['LOCALITY']} (₹{int(row['PRICE_PER_SQFT(INR)']):,}/sqft)</b><br>"
                f"<b>City Number</b> = {row['CITY_NUM']}<br>"
                f"<b>City Portion</b> = {row['CITY_LABELS']}",
    axis=1,
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Your choices")
    locality = st.selectbox(
        "Select a locality (or All)",
        ["All"] + sorted(new_df["LOCALITY"].unique().tolist()),
    )
    st.caption("Pick a locality to unlock the landmark, price and bedroom charts.")

# ---------------------------------------------------------------- header
st.title("📊 Real Estate Analytics")
st.caption("Explore Kolkata localities by price, size, landmarks and bedroom mix.")

# ---------------------------------------------------------------- KPI cards
if locality == "All":
    kpi_df = new_df
    scope = "All localities"
else:
    kpi_df = new_df[new_df["LOCALITY"] == locality]
    scope = locality

st.subheader(scope)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Listings", f"{len(kpi_df):,}")
k2.metric("Avg price", f"₹{kpi_df['PRICE_Cr'].mean():.2f} Cr")
k3.metric("Avg price / sqft", f"₹{kpi_df['PRICE_PER_SQFT(INR)'].mean():,.0f}")
k4.metric("Avg area", f"{kpi_df['AREA_SQFT'].mean():,.0f} sq ft")

# ---------------------------------------------------------------- map logic
if locality == "All":
    data_to_plot = grouped1
    center_lat = data_to_plot["LATITUDE"].mean()
    center_lon = data_to_plot["LONGITUDE"].mean()
    zoom_level = 10
    map_title = "Map of all localities in Kolkata"
else:
    data_to_plot = grouped1[grouped1["LOCALITY"] == locality]
    center_lat = data_to_plot["LATITUDE"].mean()
    center_lon = data_to_plot["LONGITUDE"].mean()
    zoom_level = 12
    map_title = f"Map of {locality}"


def build_map():
    fig = px.scatter_map(
        data_to_plot,
        lat="LATITUDE",
        lon="LONGITUDE",
        color="PRICE_PER_SQFT(INR)",
        size="AREA_SQFT",
        color_continuous_scale=px.colors.sequential.Plasma,
        zoom=zoom_level,
        map_style="open-street-map",
        center={"lat": center_lat, "lon": center_lon},
        hover_name="MAP_LABEL",
        custom_data=["LATITUDE", "LONGITUDE", "AREA_SQFT"],
    )

    fig.update_traces(
        hovertemplate=(
            "%{hovertext}<br>"
            "<b>Avg Area:</b> %{customdata[2]:,.4f} sqft<br>"
            "<b>Latitude:</b> %{customdata[0]:.6f}<br>"
            "<b>Longitude:</b> %{customdata[1]:.6f}"
            "<extra></extra>"
        ),
        marker=dict(opacity=0.7, sizemin=12, sizemode="area", symbol="circle"),
    )

    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 10, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="₹ / sqft"),
    )
    return fig



# ---------------------------------------------------------------- layout
if locality == "All":
    with st.container(border=True):
        st.markdown(f"#### {map_title}")
        st.plotly_chart(build_map(), width="stretch")
    st.info("Select a locality in the sidebar to see its landmarks, price vs area, bedroom mix and price range.")

else:
    st.markdown("#### Toggle Tabs below")
    tab_map, tab_words, tab_scatter, tab_pie, tab_box = st.tabs(
        ["🗺️ Map", "☁️ Landmarks", "📈 Price vs Area", "🥧 Bedroom Mix", "📦 Price Range"]
    )
    loc_df = new_df[new_df["LOCALITY"] == locality]

    # --- Map
    with tab_map:
        with st.container(border=True):
            st.markdown(f"#### {map_title}")
            st.plotly_chart(build_map(), width="stretch")

    # --- Word Cloud
    with tab_words:
        with st.container(border=True):
            st.markdown(f"#### Landmark details of {locality}")
            main = []
            for item in loc_df["LANDMARK_DETAILS"].dropna().apply(ast.literal_eval):
                if isinstance(item, list):
                    main.extend(item)
                elif isinstance(item, dict):
                    main.extend(item.values())
                else:
                    main.append(str(item))

            details = " ".join(main)

            if details.strip():
                wordcloud = WordCloud(
                    width=900, height=450,
                    background_color="white",
                    stopwords=STOPWORDS.union({"s"}),
                    min_font_size=3,
                ).generate(details)

                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("No landmark details available for this locality.")

    # --- Scatter plot
    with tab_scatter:
        with st.container(border=True):
            st.markdown(f"#### Scatter plot for {locality}")
            fig = px.scatter(
                loc_df, x="AREA_SQFT", y="PRICE_Cr", color="BEDROOM_NUM",
                title="Area v/s Price",
            )
            fig.update_layout(height=500, margin={"t": 50})
            st.plotly_chart(fig, width="stretch")

    # --- Pie chart
    with tab_pie:
        with st.container(border=True):
            st.markdown(f"#### Pie chart for {locality}")
            fig = px.pie(loc_df, names="BEDROOM_NUM", title="Share of listings by bedrooms")
            fig.update_layout(height=500, margin={"t": 50})
            st.plotly_chart(fig, width="stretch")

    # --- Box plot
    with tab_box:
        with st.container(border=True):
            st.markdown(f"#### Box plot for {locality}")
            fig = px.box(loc_df, x="BEDROOM_NUM", y="PRICE_Cr", title="BHK Price Range")
            fig.update_layout(height=500, margin={"t": 50})
            st.plotly_chart(fig, width="stretch")
