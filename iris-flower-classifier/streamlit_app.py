import pandas as pd
import joblib
import streamlit as st

# Load trained model
model = joblib.load("iris_model.joblib")

# Get the feature order that the model was trained on
model_columns = model.feature_names_in_

# Iris target names
target_names = ["setosa", "versicolor", "virginica"]

# Streamlit UI
st.title("🌸 Iris Flower Classifier")
st.write("Predict the species of an Iris flower based on its measurements.")
st.subheader("How to choose values")
st.write("""
    - **Setosa** has *very small petals*: petal length ~1.0-1.7
    - **Versicolor** has *medium petals*: petal length ~3.0-5.0
    - **Virginica** has *large petals*: petal length ~5.0-7.0 

    Try these example inputs: 
    - **Setosa**: 5.1, 3.5, 1.4, 0.2
    - **Versicolor**: 6.0, 2.9, 4.5, 1.5
    - **Virginica**: 6.5, 3.0, 5.5, 2.0     
""")

# User inputs
sepal_length = st.number_input(
    "Sepal length (cm)",
    0.0, 10.0, 5.0,
    help="Typical values: Setosa ~5.0, Versicolor ~6.0, Virginica ~6.5"
)
sepal_width = st.number_input(
    "Sepal width (cm)",
    0.0, 10.0, 3.5,
    help="Typical values: 3.0-3.5 for all species"
)
petal_length = st.number_input(
    "Petal length (cm)",
    0.0, 10.0, 1.5,
    help = "Key difference: Setosa ~1.5, Versicolor ~4.5, Virginica ~5.5"
)
petal_width = st.number_input(
    "Petal width (cm)",
    0.0, 10.0, 0.2,
    help="Key difference: Setosa ~0.2, Versicolor ~1.5, Virginica ~2.0"
)

# Prediction
if st.button("Predict"):
    # Force the correct column order using model.feature_names_in_
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=model_columns
    )

    prediction = model.predict(input_data)[0]
    print(prediction)
    st.success(f"The predicted Iris species is: **{target_names[prediction]}**")

    st.write("Model column order:", list(model_columns))
    st.write("Input data:", input_data)

    st.write("Model feature order:", model.feature_names_in_)
    st.write("Input order:", ["sepal_length", "sepal_width", "petal_length", "petal_width"])