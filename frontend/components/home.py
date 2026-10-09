import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------- hero
st.title("🏠 Kolkata Real Estate Intelligence")
st.markdown(
    "#### Predict property prices, explore localities and discover similar homes, "
    "all powered by machine learning."
)
st.caption("👈 Use the sidebar to open any module.")

st.divider()

# ---------------------------------------------------------------- about
st.subheader("About this app")
st.write(
    "Buying or selling a property in Kolkata means comparing hundreds of listings across "
    "dozens of localities. This app is built on a cleaned dataset of Kolkata property listings "
    "and brings three tools together in one place: a price estimator, a market explorer and a "
    "recommendation engine. Whether you want to know what a flat is worth, how a "
    "locality compares with others, or which listings are similar to one you like, you can "
    "find out here."
)

# ---------------------------------------------------------------- modules
st.subheader("What you can do")
m1, m2, m3 = st.columns(3)

with m1:
    with st.container(border=True):
        st.markdown("### 📈 Price Predictor")
        st.write(
            "Enter details such as bedrooms, area, age, floor, furnishing and location, "
            "and get an estimated price range in crores."
        )
        st.markdown(
            "- Trained regression pipeline\n"
            "- Gives a lower, expected and upper estimate\n"
            "- Takes location and building features into account"
        )

with m2:
    with st.container(border=True):
        st.markdown("### 📊 Analytics")
        st.write(
            "Explore the market locality by locality with interactive maps and charts."
        )
        st.markdown(
            "- Map of price per sqft across Kolkata\n"
            "- Landmark word cloud for each locality\n"
            "- Price vs area, bedroom mix and price range charts"
        )

with m3:
    with st.container(border=True):
        st.markdown("### 🤝 Recommender")
        st.write(
            "Pick a property you like and get similar listings, ranked by a hybrid "
            "similarity score."
        )
        st.markdown(
            "- Combines price & size, facilities & landmarks, and location\n"
            "- Adjustable weights for what matters to you\n"
            "- Map view and full details for every match"
        )

st.divider()

# ---------------------------------------------------------------- how it works
st.subheader("How it works")
h1, h2, h3 = st.columns(3)

with h1:
    with st.container(border=True):
        st.markdown("#### 1️⃣ Clean data")
        st.write(
            "Listings are cleaned and enriched with features like landmarks, "
            "furnishing, floor details and coordinates."
        )

with h2:
    with st.container(border=True):
        st.markdown("#### 2️⃣ Model & similarity")
        st.write(
            "A trained pipeline estimates prices, while cosine similarity compares "
            "properties on price, facilities and location."
        )

with h3:
    with st.container(border=True):
        st.markdown("#### 3️⃣ Interactive app")
        st.write(
            "Everything is delivered through Streamlit, so you can try inputs, "
            "change weights and see results instantly."
        )

st.divider()

# ---------------------------------------------------------------- footer
st.info(
    "💡 **Tip:** Start with the **Analytics** page to get a feel for a locality, "
    "then use the **Price Predictor** to estimate a property, and the **Recommender** "
    "to find alternatives."
)
st.caption(
    "Built with Python, Streamlit, scikit-learn, pandas and Plotly. "
    "Estimates are for guidance only and are not a formal valuation."
)