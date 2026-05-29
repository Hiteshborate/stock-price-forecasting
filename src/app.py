import streamlit as st
from advance_prediction import predict_stock

st.set_page_config(
    page_title="AI Stock Predictor",
    layout="centered"
)

st.title("📈 AI Stock Market Predictor")

st.write("Enter stock symbol like TCS.NS")

stock = st.text_input(
    "Stock Symbol",
    "TCS.NS"
)

if st.button("Predict"):

    data = predict_stock(stock)

    if "error" in data:

        st.error(data["error"])

    else:

        st.success("Prediction Generated")

        st.subheader(f"Stock: {data['stock']}")

        st.write(f"Current Price: ₹{data['current_price']}")

        st.write(f"Predicted Price: ₹{data['predicted_price']}")

        st.write(
            f"Expected Change: {data['expected_change_percent']}%"
        )

        st.write(f"Trend: {data['trend']}")

        st.write(f"Signal: {data['signal']}")