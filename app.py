import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 🧾 Streamlit UI
st.title("📈 Swing Trading AI Tool")
stock = st.text_input("📌 શેર Symbol લખો (જેમ કે TCS.NS)", "TCS.NS")

# 📥 ડેટા લોડ કરો
if stock:
    df = yf.download(stock, period="6mo", interval="1d")
    st.write("📊 છેલ્લાં 6 મહિના ના ડેટા:", df.tail())

    # 🔧 ટેક્નિકલ ઈન્ડિકેટર્સ ઉમેરો
    df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["Target"] = df["Close"].shift(-2) > df["Close"]  # 2 દિવસ પછી ભાવ વધી શકે?

    # 📘 કાચાં ફીચર્સ
    features = df[["Close", "EMA_10", "RSI"]].dropna()
    labels = df["Target"].dropna().astype(int)
    features = features.loc[labels.index]

    # 🤖 Model Train કરો
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 📈 અંતિમ દિવસ માટે પૂર્વાનુમાન
    last_data = features.tail(1)
    prediction = model.predict(last_data)[0]

    # 📋 પરિણામ
    st.subheader("📉 AI પૂર્વાનુમાન:")
    if prediction:
        st.success("✅ આ શેરમાં આવનારા 2 દિવસમાં ભાવ વધી શકે છે.")
    else:
        st.error("❌ આ શેરમાં ભાવ ઘટી શકે છે.")

    # 📊 ચાર્ટ બતાવો
    st.line_chart(df[["Close", "EMA_10"]].dropna())