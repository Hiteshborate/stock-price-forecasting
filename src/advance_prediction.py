# ============================================================
# ADVANCED AI STOCK MARKET PREDICTION SYSTEM
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import ta

from textblob import TextBlob

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ============================================================
# MAIN FUNCTION
# ============================================================

def predict_stock(stock_input):

    try:

        # ====================================================
        # STOCK SYMBOL
        # ====================================================

        stock_input = stock_input.upper()

        if ".NS" not in stock_input:

            stock = stock_input + ".NS"

        else:

            stock = stock_input

        # ====================================================
        # DOWNLOAD DATA
        # ====================================================

        data = yf.download(

            stock,

            start="2020-01-01",

            progress=False,

            auto_adjust=True
        )

        # Fix yfinance columns
        data.columns = data.columns.get_level_values(0)

        if data.empty:

            return {

                "error": "No stock data found"
            }

        # ====================================================
        # TECHNICAL INDICATORS
        # ====================================================

        close_prices = data["Close"].squeeze()

        # RSI
        data["RSI"] = ta.momentum.RSIIndicator(

            close=close_prices,

            window=14

        ).rsi()

        # MACD
        macd = ta.trend.MACD(

            close=close_prices
        )

        data["MACD"] = macd.macd()

        data["MACD_SIGNAL"] = macd.macd_signal()

        # SMA
        data["SMA_20"] = ta.trend.sma_indicator(

            close=close_prices,

            window=20
        )

        # EMA
        data["EMA_20"] = ta.trend.ema_indicator(

            close=close_prices,

            window=20
        )

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(

            close=close_prices,

            window=20
        )

        data["BB_HIGH"] = bollinger.bollinger_hband()

        data["BB_LOW"] = bollinger.bollinger_lband()

        # Momentum
        data["MOMENTUM"] = ta.momentum.ROCIndicator(

            close=close_prices,

            window=10

        ).roc()

        # Volatility
        data["VOLATILITY"] = (

            data["Close"]

            .rolling(10)

            .std()
        )

        # Returns
        data["RETURN"] = (

            data["Close"]

            .pct_change()
        )

        # Volume Change
        data["VOLUME_CHANGE"] = (

            data["Volume"]

            .pct_change()
        )

        # Price Range
        data["PRICE_RANGE"] = (

            data["High"] - data["Low"]
        )

        # ====================================================
        # TARGET
        # ====================================================

        data["TARGET"] = data["Close"].shift(-1)

        # ====================================================
        # CLEAN DATA
        # ====================================================

        data.replace(

            [np.inf, -np.inf],

            np.nan,

            inplace=True
        )

        data.dropna(inplace=True)

        # ====================================================
        # FEATURES
        # ====================================================

        features = [

            "Open",
            "High",
            "Low",
            "Volume",

            "RSI",

            "MACD",
            "MACD_SIGNAL",

            "SMA_20",
            "EMA_20",

            "BB_HIGH",
            "BB_LOW",

            "MOMENTUM",

            "VOLATILITY",

            "RETURN",

            "VOLUME_CHANGE",

            "PRICE_RANGE"
        ]

        X = data[features]

        y = data["TARGET"]

        # ====================================================
        # TRAIN TEST SPLIT
        # ====================================================

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.2,

            shuffle=False,

            random_state=42
        )

        # ====================================================
        # MACHINE LEARNING MODEL
        # ====================================================

        model = RandomForestRegressor(

            n_estimators=300,

            max_depth=15,

            min_samples_split=5,

            min_samples_leaf=2,

            random_state=42,

            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # ====================================================
        # NEXT DAY PREDICTION
        # ====================================================

        latest_data = data[features].iloc[-1:]

        predicted_price = float(

            model.predict(latest_data)[0]
        )

        # ====================================================
        # CURRENT PRICE
        # ====================================================

        current_price = float(

            data["Close"].iloc[-1]
        )

        # ====================================================
        # EXPECTED CHANGE
        # ====================================================

        expected_change = (

            (predicted_price - current_price)

            / current_price

        ) * 100

        # ====================================================
        # SENTIMENT ANALYSIS
        # ====================================================

        sample_news = [

            f"{stock_input} reports strong quarterly earnings",

            f"Investors are optimistic about {stock_input}",

            f"{stock_input} market outlook remains positive",

            f"Analysts expect growth in {stock_input}",

            f"{stock_input} stock gains investor attention"
        ]

        sentiment_scores = []

        for news in sample_news:

            analysis = TextBlob(news)

            sentiment_scores.append(

                analysis.sentiment.polarity
            )

        avg_sentiment = np.mean(sentiment_scores)

        # ====================================================
        # TECHNICAL VALUES
        # ====================================================

        latest_rsi = float(
            data["RSI"].iloc[-1]
        )

        latest_macd = float(
            data["MACD"].iloc[-1]
        )

        latest_momentum = float(
            data["MOMENTUM"].iloc[-1]
        )

        latest_sma = float(
            data["SMA_20"].iloc[-1]
        )

        latest_ema = float(
            data["EMA_20"].iloc[-1]
        )

        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        r2 = r2_score(

            y_test,
            predictions
        )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = 50

        if r2 > 0.90:

            confidence += 20

        elif r2 > 0.80:

            confidence += 15

        elif r2 > 0.70:

            confidence += 10

        confidence += abs(avg_sentiment) * 20

        if latest_momentum > 0:

            confidence += 10

        confidence = max(
            0,
            min(100, confidence)
        )

        # ====================================================
        # TREND LOGIC
        # ====================================================

        bullish = 0
        bearish = 0

        # Price Prediction
        if predicted_price > current_price:

            bullish += 3

        else:

            bearish += 3

        # RSI
        if latest_rsi >= 60:

            bullish += 2

        elif latest_rsi >= 45:

            bullish += 1

        elif latest_rsi < 40:

            bearish += 2

        # MACD
        if latest_macd > 0:

            bullish += 2

        else:

            bearish += 2

        # Sentiment
        if avg_sentiment > 0.15:

            bullish += 2

        elif avg_sentiment > 0:

            bullish += 1

        elif avg_sentiment < -0.15:

            bearish += 2

        # Momentum
        if latest_momentum > 0:

            bullish += 2

        else:

            bearish += 2

        # SMA
        if current_price > latest_sma:

            bullish += 1

        else:

            bearish += 1

        # EMA
        if current_price > latest_ema:

            bullish += 1

        else:

            bearish += 1

        # ====================================================
        # FINAL TREND
        # ====================================================

        difference = bullish - bearish

        if difference >= 3:

            trend = "BULLISH"

            if confidence >= 75:

                signal = "STRONG BUY"

            else:

                signal = "BUY"

        elif difference <= -3:

            trend = "BEARISH"

            if confidence >= 75:

                signal = "STRONG SELL"

            else:

                signal = "SELL"

        else:

            trend = "NEUTRAL"

            signal = "HOLD"

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {

            "stock": stock,

            "current_price": round(current_price, 2),

            "predicted_price": round(predicted_price, 2),

            "expected_change_percent": round(expected_change, 2),

            "trend": trend,

            "signal": signal,

            "confidence": round(confidence, 2)
        }

    except Exception as e:

        return {

            "error": str(e)
        }