
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import os

st.set_page_config(
    page_title="Starbucks Decision Tree",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Decision Tree Predictor")
st.write("Train a Decision Tree model and predict beverage calories.")

# Load dataset
uploaded_file = st.sidebar.file_uploader(
    "Upload Starbucks CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "starbucks.csv"
    )

    if not os.path.exists(file_path):
        st.info("Upload your Starbucks CSV file from the sidebar.")
        st.stop()

    df = pd.read_csv(file_path)

df.columns = df.columns.astype(str).str.strip()
df = df.drop_duplicates()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Select target
numeric_columns = df.select_dtypes(include="number").columns.tolist()

if not numeric_columns:
    st.error("No numeric columns found in your dataset.")
    st.stop()

default_target = next(
    (c for c in numeric_columns if c.lower() == "calories"),
    numeric_columns[0]
)

target = st.selectbox(
    "Select target to predict",
    numeric_columns,
    index=numeric_columns.index(default_target)
)

# Prepare data
data = df.copy()
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target])

features = [
    c for c in data.columns
    if c != target and data[c].notna().any()
]

if len(data) < 10:
    st.error("Please use a dataset with at least 10 valid rows.")
    st.stop()

if not features:
    st.error("No usable input features found.")
    st.stop()

X = data[features].copy()
y = data[target].copy()

categorical_columns = X.select_dtypes(
    include=["object", "string", "category", "bool"]
).columns.tolist()

numeric_features = [
    c for c in X.columns if c not in categorical_columns
]

transformers = []

if numeric_features:
    transformers.append((
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numeric_features
    ))

if categorical_columns:
    transformers.append((
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_columns
    ))

preprocessor = ColumnTransformer(transformers=transformers)

# Decision Tree settings
st.subheader("Decision Tree Settings")

max_depth = st.slider("Maximum Tree Depth", 1, 20, 5)
min_samples_leaf = st.slider("Minimum Samples per Leaf", 1, 10, 2)
test_percentage = st.slider("Test Data (%)", 10, 40, 20, 5)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("tree", DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
])

test_rows = max(2, round(len(X) * test_percentage / 100))
test_rows = min(test_rows, len(X) - 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_rows,
    random_state=42
)

if st.button("Train Decision Tree", type="primary"):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    st.session_state["model"] = model
    st.session_state["features"] = features
    st.session_state["target"] = target
    st.session_state["numeric_features"] = numeric_features
    st.session_state["categorical_features"] = categorical_columns
    st.session_state["training_columns"] = X.columns.tolist()

    st.success("Decision Tree trained successfully!")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training Rows",
        len(X_train)
    )

    col2.metric(
        "Mean Absolute Error",
        f"{mean_absolute_error(y_test, predictions):.2f}"
    )

    score = r2_score(y_test, predictions)
    col3.metric("R² Score", f"{score:.3f}")

    st.subheader("Actual vs Predicted Values")

    results = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions
    })

    st.dataframe(results, use_container_width=True)

# Prediction interface
st.divider()
st.header("Predict Starbucks Beverage Calories")

if "model" not in st.session_state:
    st.info("Click Train Decision Tree before making a prediction.")

elif st.session_state["target"] != target:
    st.warning("The target has changed. Train the model again.")

else:
    saved_features = st.session_state["features"]
    numeric_set = st.session_state["numeric_features"]

    with st.form("prediction_form"):
        user_values = {}

        for column in saved_features:
            if column in numeric_set:
                values = pd.to_numeric(
                    df[column], errors="coerce"
                ).dropna()

                default = float(values.median()) if not values.empty else 0.0

                user_values[column] = st.number_input(
                    column,
                    value=default,
                    key=f"input_{column}"
                )
            else:
                options = sorted(
                    df[column].dropna().astype(str).unique().tolist()
                )

                if options:
                    user_values[column] = st.selectbox(
                        column,
                        options,
                        key=f"input_{column}"
                    )
                else:
                    user_values[column] = ""

        predict_clicked = st.form_submit_button("Predict Calories")

    if predict_clicked:
        input_df = pd.DataFrame(
            [user_values],
            columns=saved_features
        )

        prediction = st.session_state["model"].predict(input_df)[0]

        st.success(
            f"Predicted {target}: {prediction:.2f}"
        )

        st.caption(
            "This is an estimate from your trained Decision Tree model."
        )
