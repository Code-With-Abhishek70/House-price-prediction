import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "house_model.pkl"
DATA_PATH = "HousePricePrediction.csv"
CAT_COLS = ["MSZoning", "LotConfig", "BldgType", "Exterior1st"]


@st.cache_resource
def load_everything():
    """Notebook jaisa hi preprocessing dobara karke scaler + columns banata hai."""
    model = joblib.load(MODEL_PATH)

    raw = pd.read_csv(DATA_PATH)
    df = raw.drop(columns=["Id"])
    df["SalePrice"] = df["SalePrice"].fillna(df["SalePrice"].mean())
    df = df.dropna()

    # dropdown ke options (encoding se pehle wali values)
    options = {c: sorted(df[c].unique().tolist()) for c in CAT_COLS}
    num_stats = df.drop(columns=CAT_COLS + ["SalePrice"]).describe().loc[["min", "max", "50%"]]

    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler().fit(X_train)
    return model, scaler, list(X.columns), options, num_stats


def build_row(inputs, columns):
    """User ke inputs ko training jaise 33 columns me badalta hai."""
    row = pd.DataFrame([inputs])
    row = pd.get_dummies(row, columns=CAT_COLS)
    return row.reindex(columns=columns, fill_value=0).astype(float)


st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction")

try:
    model, scaler, columns, options, stats = load_everything()
except FileNotFoundError as e:
    st.error(f"File nahi mili: {e.filename}. `app.py` ke folder me "
             f"`{MODEL_PATH}` aur `{DATA_PATH}` rakho.")
    st.stop()

st.subheader("Fill in the details of the house to predict its price")
left, right = st.columns(2)

inputs = {}
inputs["MSSubClass"] = left.number_input("MSSubClass (building class)", 20, 190, 60, step=10)
inputs["LotArea"] = right.number_input("Lot Area (sq ft)", 1000, 200000, 8500, step=100)
inputs["OverallCond"] = left.slider("Overall Condition (1-9)", 1, 9, 5)
inputs["YearBuilt"] = right.number_input("Year Built", 1870, 2025, 2000)
inputs["YearRemodAdd"] = left.number_input("Year Remodelled", 1950, 2025, 2000)
inputs["BsmtFinSF2"] = right.number_input("Basement Finished SF (type 2)", 0, 2000, 0, step=10)
inputs["TotalBsmtSF"] = left.number_input("Total Basement SF", 0, 6000, 1000, step=50)

inputs["MSZoning"] = right.selectbox("Zoning", options["MSZoning"])
inputs["LotConfig"] = left.selectbox("Lot Configuration", options["LotConfig"])
inputs["BldgType"] = right.selectbox("Building Type", options["BldgType"])
inputs["Exterior1st"] = left.selectbox("Exterior Material", options["Exterior1st"])

if st.button("Predict Price", type="primary"):
    X_row = build_row(inputs, columns)
    prediction = float(model.predict(scaler.transform(X_row))[0])
    st.success(f"Estimated Price: ${prediction:,.0f}")

    result = pd.DataFrame([{**inputs, "predicted_price": round(prediction, 2)}])
    st.download_button("Download result (CSV)", result.to_csv(index=False),
                       file_name="prediction.csv", mime="text/csv")
