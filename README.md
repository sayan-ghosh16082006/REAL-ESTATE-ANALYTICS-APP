# 🏠 Kolkata Real Estate Intelligence

A multi-page **Streamlit** web app that brings three machine-learning tools together for the Kolkata property market:

| Module | What it does |
|---|---|
| 📈 **Price Predictor** | Estimates a property's price range (in ₹ crore) from its features |
| 📊 **Analytics** | Explores localities with maps, word clouds and price/size/bedroom charts |
| 🤝 **Recommender** | Finds similar properties using a hybrid similarity score |

> Estimates are for guidance only and are not a formal valuation.

---

## ✨ Features

### 📈 Price Predictor
- Inputs: bedrooms, balconies, area, age, floors, landmark count, cooperative society, facing, furnishing, property type, city, locality and coordinates.
- A trained scikit-learn pipeline predicts the price. The model works on a log scale, so predictions are converted back with `np.expm1`.
- Shows a **lower, expected and upper** estimate (the model estimate ±10%).

### 📊 Analytics
- Interactive map of all localities, coloured by average price per sq ft and sized by average area.
- Per-locality tabs: map, **landmark word cloud**, price vs area scatter plot, bedroom mix pie chart, and price range box plot.
- KPI cards for the selected scope: listings, average price, price per sq ft and area.

### 🤝 Recommender
Pick a property and get similar ones, ranked by a **hybrid score** built from three signals:

1. **Price & size**: AGE, BEDROOM_NUM, BALCONY_NUM, AREA_SQFT and PRICE_Cr.
2. **Facilities & landmarks**: TF-IDF on descriptions, tags, landmarks, furnishing and property type.
3. **Location**: similarity based on latitude and longitude.

Each signal is a precomputed cosine similarity matrix. The user sets the weights with sliders, and the final score is the weighted sum, normalised to 0–1. Results appear on a map (selected property in red, matches in blue), with a **View details** panel for any match.

---

## 🗂️ Project structure

```
realestate-app/
├── backend/                         # model training & similarity-matrix code (notebooks)
├── frontend/
│   ├── main.py                      # app entry point (navigation)
│   ├── components/
│   │   ├── home.py                  # landing page
│   │   ├── predictor.py             # price prediction
│   │   ├── analytics.py             # maps & charts
│   │   └── recommender.py           # hybrid recommender
│   ├── pkl_file/
│   │   ├── pipeline.pkl             # trained price model
│   │   ├── df.pkl                   # dataframe used for dropdown options
│   │   ├── cosine_sim1.pkl          # facilities / landmarks similarity
│   │   ├── cosine_sim_price.pkl     # price & size similarity
│   │   └── cosine_sim_location.pkl  # location similarity
│   └── datasets/
│       └── kolkata_cleaned_v5.csv   # cleaned listings
└── pyproject.toml
```

> The page file names above are examples. The pages locate `pkl/` and `datasets/` with `Path(__file__).parent.parent`, so they must sit one folder below the project root.

---

## 🚀 Getting started

### 1. Clone and create an environment
```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run Home.py
```
Then open the URL Streamlit prints (usually http://localhost:8501).

---

## 📦 Requirements

Create a `requirements.txt` with:

```
streamlit
pandas
numpy
scikit-learn
plotly>=5.24
matplotlib
wordcloud
pydeck
```

`plotly>=5.24` is needed for `px.scatter_map`. The `scikit-learn` version should match the one used to train and pickle the model, otherwise loading `pipeline.pkl` may fail or warn.

---

## 🛠️ How the recommender is built

The similarity matrices are created in the notebook and saved as pickles:

```python
import pickle

pickle.dump(cosine_sim1, open("pkl/cosine_sim1.pkl", "wb"))
pickle.dump(cosine_sim_price, open("pkl/cosine_sim_price.pkl", "wb"))
pickle.dump(cosine_sim_location, open("pkl/cosine_sim_location.pkl", "wb"))
```

**Important:** each matrix is indexed by row position, so the CSV loaded by the app must have **exactly the same rows in the same order** as the dataframe used to create them. The Recommender page checks this and shows an error if the sizes differ. If you clean, filter or de-duplicate the data later, rebuild and re-save the matrices.

---

## ⚠️ Known limitations

- Predictions depend on the training data, so unusual properties or new localities can be less accurate.
- The ±10% price range is a fixed margin, not a statistical confidence interval.
- Some source rows had latitude and longitude swapped. The app corrects them for display, but the pickled location matrix reflects whatever data it was built from.
- Recommendations don't update until **Recommend similar properties** is clicked again after changing a setting.

---

## 🧰 Tech stack

Python · Streamlit · pandas · NumPy · scikit-learn · Plotly · pydeck · Matplotlib · WordCloud

---

## 📄 License

Add your license here (for example MIT).