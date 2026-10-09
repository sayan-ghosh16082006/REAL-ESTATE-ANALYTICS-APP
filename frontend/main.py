import streamlit as st
from pathlib import Path

Basepath = Path(__file__).parent / "components"

predictor_path = Basepath/"price_predictor.py"
analytics_path = Basepath/"analytics_module.py"
home_path = Basepath/"home.py"
recommender_path = Basepath / "recommender.py"

home = st.Page(home_path, title="Home Page")
price_predictor_component = st.Page(predictor_path, title="Price Predictor")
analytics_component = st.Page(analytics_path, title="Analytics")
recommender_component = st.Page(recommender_path, title="Recommender")

pg = st.navigation(pages=[home,price_predictor_component, analytics_component, recommender_component])



pg.run()


