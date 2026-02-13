import os
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===============================
# CONFIG
# ===============================
SYMBOL = "BTC-USD"
INTERVAL = "5m"
PERIOD = "7d"
VOL_WINDOW = 20

os.makedirs("out", exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
ticker = yf.Ticker(SYMBOL)
df = ticker.history(period=PERIOD, interval=INTERVAL)

if df.empty:
    raise RuntimeError("No data received from yfinance")

# ===============================
# RETURNS
# ===============================
df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

returns = df["ret"] * 100  # scaling matters

# ===============================
# EGARCH(1,1) MODEL
# ===============================
model = arch_model(
    returns,
    vol="EGARCH",
    p=1,
    q=1,
    mean="Zero",
    dist="StudentsT"
)

res = model.fit(disp="off")

df["egarch_vol"] = res.conditional_volatility / 100
df["vol_mean"] = df["egarch_vol"].rolling(VOL_WINDOW).mean()
df["shock"] = df["egarch_vol"] > 2.0 * df["vol_mean"]

# ===============================
# TERMINAL OUTPUT
# ===============================
print("\nEGARCH(1,1) SUMMARY")
print("----------------------------")
print(f"Symbol      : {SYMBOL}")
print(f"EGARCH Vol  : {df['egarch_vol'].iloc[-1]:.6f}")
print(f"Shock       : {bool(df['shock'].iloc[-1])}")
print("\nModel Params")
print(res.params)

# ===============================
# PLOTLY: PRICE + EGARCH VOL
# ===============================
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=("Price Action (Close)", "EGARCH(1,1) Volatility")
)

# ---- PRICE (TOP) ----
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="black")
    ),
    row=1,
    col=1
)

# ---- VOLATILITY (BOTTOM) ----
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["egarch_vol"],
        mode="lines",
        name="EGARCH Volatility",
        line=dict(color="purple")
    ),
    row=2,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["vol_mean"],
        mode="lines",
        name="Volatility Mean",
        line=dict(color="blue", dash="dash")
    ),
    row=2,
    col=1
)

# ---- SHOCK MARKERS ----
fig.add_trace(
    go.Scatter(
        x=df.index[df["shock"]],
        y=df.loc[df["shock"], "egarch_vol"],
        mode="markers",
        name="Shock",
        marker=dict(color="orange", size=6)
    ),
    row=2,
    col=1
)

# ---- LAYOUT ----
fig.update_layout(
    title=f"{SYMBOL} — Price Action & EGARCH(1,1) Volatility",
    template="plotly_white",
    height=850,
    legend=dict(orientation="h", y=1.03)
)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Volatility", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=1)

# ===============================
# SAVE INTERACTIVE HTML
# ===============================
OUTPUT_FILE = "out/egarch_price_vol.html"
fig.write_html(OUTPUT_FILE)

print(f"\nInteractive plot saved → {OUTPUT_FILE}")
