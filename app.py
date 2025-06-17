import streamlit as st
import joblib

# Load the saved model
model = joblib.load("genre_model.pkl")

# Streamlit page config
st.set_page_config(page_title="Movie Genre Predictor")
st.title("🎬 Movie Genre Prediction App")
st.markdown("Enter a short movie description below:")

# Text input box
description = st.text_area("Movie Description", height=150)

# Prediction button
if st.button("Predict Genre"):
    if description.strip() == "":
        st.warning("Please enter a movie description to predict.")
    else:
        # Predict genre
        predicted_genre = model.predict([description])[0]
        st.success(f"Predicted Genre: **{predicted_genre}** 🎯")
