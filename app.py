import streamlit as st
import joblib
import pandas as pd
import json

page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("fondo.jpg");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
# Add a new page for Power BI chart
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Price", "Power BI Chart"])

if page == "Power BI Chart":
    st.title("Power BI Chart")
    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiN2U0NDQyOWQtNzg3Yy00N2Q5LWIyZDQtN2FhZWZmMmM1YWRjIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9"
    st.components.v1.iframe(powerbi_url, height=600)

else:

    # Load the trained model and scalers
    model = joblib.load(r'./price_predictor_model.pkl')
    feature_scaler = joblib.load(r'./feature_scaler.pkl')
    target_scaler = joblib.load(r'./target_scaler.pkl')

    with open(r'./data/jsons/room_type_encoding.json', 'r') as f:
        room_type_dict = json.load(f)
        
    # Define the features used in the model
    features = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'room_type_encoded', 'review_scores_rating']

    # Streamlit app
    st.title("House/Flat Price Predictor")

    # Input fields
    accommodates = st.number_input("Accommodates", min_value=1, max_value=20, value=1)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)
    beds = st.number_input("Beds", min_value=0, max_value=20, value=1)
    room_type = st.selectbox("Room Type", ['Private room', 'Entire home/apt', 'Hotel room', 'Shared room'])
    review_scores_rating = st.number_input("Review Scores Rating", min_value=0.0, max_value=5.0, value=4.0)


    # Predict button
    if st.button("Predict Price"):
        room_type_encoded = room_type_dict[room_type]

        # Create input data
        input_data = pd.DataFrame([{
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'beds': beds,
            'room_type_encoded':room_type_encoded,
            'review_scores_rating': review_scores_rating
        }])
        
        # Scale the numerical variables using the feature scaler
        numerical_features = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'review_scores_rating']
        input_data = pd.DataFrame(feature_scaler.transform(input_data))
        

        # Predict the price
        prediction_scaled = model.predict(input_data)

        # Inverse transform the prediction
        prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        # Display the prediction
        st.success(f"The predicted price is ${prediction:.2f}")

