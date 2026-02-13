
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go


# ---------------------------------
# NIFTY 50 Stocks (Hardcoded - No Scraping)
# ---------------------------------

tickers = [
    "^NSEI",
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BEL.NS","BHARTIARTL.NS",
    "BPCL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DRREDDY.NS",
    "EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS",
    "INFY.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS",
    "M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS",
    "POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SHRIRAMFIN.NS",
    "SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS","TCS.NS",
    "TECHM.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS"
]

for ticker in tickers:

    print(f"Processing {ticker}...")

    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    # Fix MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df is None or df.empty:
        print(f"Download failed for {ticker}")
        continue

    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    # ---------------------------------
    # Add Downtrend Filter (MA Slope)
    # ---------------------------------
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA20_slope'] = df['MA20'].diff()

    # ---------------------------------
    # Detect Local Minima
    # ---------------------------------
    order = 5
    prices = df['Low'].values
    local_mins_idx = argrelextrema(prices, np.less_equal, order=order)[0]

    # ---------------------------------
    # Double Bottom Detection
    # ---------------------------------
    tolerance = 0.03
    min_separation = 10
    patterns = []

    for i in range(len(local_mins_idx) - 1):
        t1 = local_mins_idx[i]
        t2 = local_mins_idx[i + 1]

        if t2 - t1 < min_separation:
            continue

        if df['MA20_slope'].iloc[t1] >= 0:
            continue

        price1 = df['Low'].iloc[t1]
        price2 = df['Low'].iloc[t2]

        if abs(price1 - price2) / price1 > tolerance:
            continue

        middle_high = df['High'].iloc[t1:t2]
        if len(middle_high) == 0:
            continue

        neckline = middle_high.max()

        breakout_idx = None
        for j in range(t2 + 1, len(df)):
            if df['Close'].iloc[j] > neckline:
                breakout_idx = j
                break

        if breakout_idx is not None:
            patterns.append({
                "t1": t1,
                "t2": t2,
                "breakout_idx": breakout_idx,
                "neckline": neckline
            })

    # ---------------------------------
    # Plot
    # ---------------------------------
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    ))

    for p in patterns:
        fig.add_trace(go.Scatter(
            x=[df['Date'].iloc[p['t1']]],
            y=[df['Low'].iloc[p['t1']]],
            mode="markers",
            marker=dict(size=10, symbol="circle"),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[df['Date'].iloc[p['t2']]],
            y=[df['Low'].iloc[p['t2']]],
            mode="markers",
            marker=dict(size=10, symbol="circle"),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[df['Date'].iloc[p['t1']], df['Date'].iloc[p['breakout_idx']]],
            y=[p['neckline'], p['neckline']],
            mode="lines",
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[df['Date'].iloc[p['breakout_idx']]],
            y=[df['Close'].iloc[p['breakout_idx']]],
            mode="markers",
            marker=dict(size=12, symbol="triangle-up"),
            showlegend=False
        ))

    fig.update_layout(
        title=f"{ticker} Double Bottom Detection",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    filename = f"{ticker.replace('^','')}_double_bottom.html"
    fig.write_html(filename)

    print(f"Saved chart: {filename}")