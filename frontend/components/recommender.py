import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import requests

st.set_page_config(page_title="Kolkata Property Recommender", page_icon="🏠", layout="wide")

BASE = Path(__file__).parent.parent
DATA_PATH = BASE / "datasets" / "kolkata_cleaned_v5.csv"

# HuggingFace URLs
URLS = {
    "cosine_sim1": "https://huggingface.co/sayan2006/realestate-models/resolve/main/cosine_sim1.pkl",
    "cosine_sim_price": "https://huggingface.co/sayan2006/realestate-models/resolve/main/cosine_sim_price.pkl",
    "cosine_sim_location": "https://huggingface.co/sayan2006/realestate-models/resolve/main/cosine_sim_location.pkl",
}


RED, BLUE, GREEN = [230, 57, 70], [29, 78, 216], [22, 163, 74]
MAP_STYLES = {
    "Light":   "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Dark":    "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "Streets": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
}

# ---------------------------------------------------------------- helpers
def extract_list(val):
    """Turn a stringified list like "['ATM', 'Hospital']" into a real list."""
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, SyntaxError):
        return []


# ---------------------------------------------------------------- loading (cached)
@st.cache_data(show_spinner="Loading data...")
def load_data():
    df = pd.read_csv(DATA_PATH).reset_index(drop=True)

    # Fix rows where latitude/longitude are swapped (map display only)
    swapped = df["LATITUDE"] > 60
    df.loc[swapped, ["LATITUDE", "LONGITUDE"]] = df.loc[swapped, ["LONGITUDE", "LATITUDE"]].values

    # Readable label (does NOT need to be unique, the dropdown returns the row index)
    df["LABEL"] = (
        df["PROP_NAME"].astype(str).str.title()
        + "  |  " + df["PROP_HEADING"].astype(str)
        + "  |  ₹" + df["PRICE_Cr"].round(2).astype(str) + " Cr"
    )
    return df


@st.cache_resource(show_spinner="Loading similarity matrices...")
def load_matrices():
    def _load(url):
        response = requests.get(url)
        return np.asarray(pickle.loads(response.content))  # load directly from bytes

    return (
        _load(URLS["cosine_sim1"]), 
        _load(URLS["cosine_sim_price"]), 
        _load(URLS["cosine_sim_location"])
    )


df = load_data()
sim_text, sim_price, sim_loc = load_matrices()

# Matrices must line up row-for-row with the CSV
if not (sim_text.shape[0] == sim_price.shape[0] == sim_loc.shape[0] == len(df)):
    st.error(
        f"Row mismatch: CSV has {len(df)} rows but matrices have "
        f"{sim_text.shape[0]}/{sim_price.shape[0]}/{sim_loc.shape[0]}. "
        "Re-save the pickles from the same dataframe you use here."
    )
    st.stop()


# ---------------------------------------------------------------- recommender
def hybrid_recommend_properties_by_location(idx, w_price, w_landmark, w_location, top_n, city="All"):
    price_scores = sim_price[idx]
    text_scores = sim_text[idx]
    loc_scores = sim_loc[idx]

    total = (w_price + w_landmark + w_location) or 1  # normalise so Score stays in 0-1
    final = (w_price * price_scores + w_landmark * text_scores + w_location * loc_scores) / total

    out = df.copy()
    out["Score"] = final
    out["Text"] = text_scores
    out["Price/Size"] = price_scores
    out["Location"] = loc_scores

    out = out.drop(index=idx)  # exclude the selected property itself
    if city != "All":
        out = out[out["CITY_LABELS"] == city]

    return out.sort_values("Score", ascending=False).head(top_n)


# ---------------------------------------------------------------- maps
def build_map_df(src_row, recs, detail_idx):
    """
    No detail open  -> selected property RED, recommendations BLUE.
    Detail open     -> viewed property RED, original selection GREEN, others BLUE.
    'hi' = True marks the red point that must be drawn on top, bigger.
    """
    rows = []

    for row_idx, r in recs.iterrows():
        if row_idx != detail_idx:
            rows.append((r.LATITUDE, r.LONGITUDE, BLUE, str(r.PROP_NAME).title(), False))

    if detail_idx is None:
        rows.append((src_row.LATITUDE, src_row.LONGITUDE, RED,
                     "Selected: " + str(src_row.PROP_NAME).title(), True))
    else:
        rows.append((src_row.LATITUDE, src_row.LONGITUDE, GREEN,
                     "Your selection: " + str(src_row.PROP_NAME).title(), False))
        d = recs.loc[detail_idx]
        rows.append((d.LATITUDE, d.LONGITUDE, RED, "Viewing: " + str(d.PROP_NAME).title(), True))

    return pd.DataFrame(rows, columns=["lat", "lon", "color", "name", "hi"])


def _layer(data, min_px, max_px):
    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[lon, lat]",
        get_fill_color="color",
        auto_highlight=True,
        get_radius=20,                 # metres, small on purpose
        radius_min_pixels=min_px,      # never smaller than this on screen
        radius_max_pixels=max_px,      # never bigger than this on screen
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        pickable=True,
    )


def draw_map(map_df, zoom=None, height=450):
    base = map_df[~map_df["hi"]]
    top = map_df[map_df["hi"]]

    layers = []
    if not base.empty:
        layers.append(_layer(base, 6, 9))     # blue / green (bottom layer)
    if not top.empty:
        layers.append(_layer(top, 10, 13))    # red (top layer, always drawn last)

    if zoom is not None or len(map_df) == 1:
        view = pdk.ViewState(
            latitude=float(map_df["lat"].mean()),
            longitude=float(map_df["lon"].mean()),
            zoom=zoom or 14,
        )
    else:
        view = pdk.data_utils.compute_view(map_df[["lon", "lat"]].values.tolist(), 0.9)

    st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={"text": "{name}"},
        map_style=MAP_STYLES[st.session_state.get("map_style", "Light")],
    ),
    height=height,
    )


# ---------------------------------------------------------------- UI
st.title("🏠 Kolkata Property Recommender")
st.caption("Pick a property you like and get similar ones, ranked by facilities, price/size and location.")

with st.sidebar:
    st.header("Your choices")
    idx = st.selectbox(
        "Select a property",
        options=df.index.tolist(),                  # returns the row index directly
        format_func=lambda i: df.at[i, "LABEL"],    # shows the readable label
    )
    top_n = st.slider("How many recommendations", 3, 20, 10)
    city = st.selectbox("Limit to area", ["All"] + sorted(df["CITY_LABELS"].dropna().unique()))

    st.subheader("What matters to you?")
    w_price = st.slider("Price & size", 0.0, 1.0, 0.4, 0.05)
    w_landmark = st.slider("Facilities & landmarks", 0.0, 1.0, 0.3, 0.05)
    w_loc = st.slider("Location", 0.0, 1.0, 0.3, 0.05)

src = df.loc[idx]

st.subheader(f"Selected property : {(src.PROP_NAME).capitalize()}")
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 2, 1.5])
c1.metric("Price", f"₹{src.PRICE_Cr:.2f} Cr")
c2.metric("Area", f"{src.AREA_SQFT:.0f} sq ft")
c3.metric("Bedrooms", f"{src.BEDROOM_NUM:.0f}")
c4.metric("Property Type", f"{src.PROPERTY_type}")
c5.metric("Locality", src.LOCALITY)
with st.expander("Description"):
    st.write(src.DESCRIPTION)

# ---- 1) Run the recommender and STORE the result
if st.button("Recommend similar properties", type="primary"):
    st.session_state.recs = hybrid_recommend_properties_by_location(
        idx, w_price, w_landmark, w_loc, top_n, city
    )
    st.session_state.src_idx = idx        # remember which property these belong to
    st.session_state.detail_idx = None    # clear any previously opened detail

# ---- 2) Show results from session_state (survives reruns)
recs = st.session_state.get("recs")

if recs is not None:
    if recs.empty:
        st.warning("No properties found for these filters.")
    else:
        src_row = df.loc[st.session_state.src_idx]
        detail_idx = st.session_state.get("detail_idx")   # read BEFORE drawing the map

        # ---------- (A) Overview map
        st.subheader(f"Top {len(recs)} recommendations")
        draw_map(build_map_df(src_row, recs, detail_idx))
        if detail_idx is None:
            st.caption("🔴 Selected property   🔵 Recommendations")
        else:
            st.caption("🔴 Property being viewed   🟢 Your original selection   🔵 Other recommendations")

        # ---------- (B) Detail panel, directly BELOW the overview map
        if detail_idx is not None:
            d = df.loc[detail_idx]
            with st.container(border=True):
                top_l, top_r = st.columns([6, 1])
                top_l.subheader(f"{str(d.PROP_NAME).title()} — {d.PROP_HEADING}")
                if top_r.button("✖ Close"):
                    st.session_state.detail_idx = None
                    st.rerun()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"₹{d.PRICE_Cr:.2f} Cr")
                m2.metric("Area", f"{d.AREA_SQFT:.0f} sq ft")
                m3.metric("Bedrooms", f"{d.BEDROOM_NUM:.0f}")
                m4.metric("Balconies", f"{d.BALCONY_NUM:.0f}")

                t1, t2, t3, t4 = st.columns(4)
                t1.markdown(f"**Property Type**  \n{d.PROPERTY_type}")
                t2.markdown(f"**Locality**  \n{d.LOCALITY}")
                t3.markdown(f"**Furnishing**  \n{d.FURNISH_LABEL}")
                t4.markdown(f"**Age**  \n{d.AGE:.0f} yrs")

                st.markdown("**Description**")
                st.write(d.DESCRIPTION)

                tags = extract_list(d.SECONDARY_TAGS)
                if tags:
                    st.markdown("**Highlights:** " + " · ".join(map(str, tags)))

                landmarks = extract_list(d.LANDMARK_DETAILS)
                if landmarks:
                    st.markdown("**Nearby landmarks**")
                    for lm in landmarks:
                        st.markdown(f"- {lm}")

                # Zoomed-in map of just this property (red)
                draw_map(
                    pd.DataFrame([(d.LATITUDE, d.LONGITUDE, RED, str(d.PROP_NAME).title(), True)],
                                 columns=["lat", "lon", "color", "name", "hi"]),
                    zoom=14, height=300,
                )

                with st.expander("All raw fields"):
                    st.dataframe(d.astype(str).rename("Value"), width = "stretch")

        # ---------- (C) Cards, each with a "View details" button
        for row_idx, r in recs.iterrows():
            with st.container(border=True):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(f"**{((str(r.PROP_NAME)).title()).strip()}** — {r.PROP_HEADING}")
                    st.write(
                        f"₹{r.PRICE_Cr:.2f} Cr · {r.AREA_SQFT:.0f} sq ft · "
                        f"{r.BEDROOM_NUM:.0f} BHK · {r.FURNISH_LABEL} · {r.CITY_LABELS}"
                    )
                    st.caption(str(r.DESCRIPTION)[:220] + "...")
                with right:
                    st.metric("Match", f"{r.Score * 100:.0f}%")
                    st.caption(
                        f"Facilities {r.Text:.2f} · Price {r['Price/Size']:.2f} · Location {r.Location:.2f}"
                    )
                    if st.button("View details", key=f"detail_{row_idx}"):
                        st.session_state.detail_idx = row_idx
                        st.rerun()