import pandas as pd
import numpy as np
import easyquotation
import time
import os
from datetime import datetime, timedelta

from model import Kronos, KronosTokenizer, KronosPredictor

# ========== CONFIG ==========
STOCK_CODE = "002877"   # example stock
INTERVAL = 300          # 5 min in seconds
LOOKBACK = 240          # number of past 5-min bars (20 hours)
PRED_LEN = 12           # predict next 12 steps = 1 hour
DEVICE = "cuda:0"
DATA_DIR = "./data"
BAR_FILE = os.path.join(DATA_DIR, "realtime_bars.csv")
PRED_FILE = os.path.join(DATA_DIR, "predictions.csv")
# ============================

# ensure data dir exists
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Load Model
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)

# 2. Init easyquotation
q = easyquotation.use("sina")

# 3. K-line storage
bars = []  # each bar: dict {timestamp, open, high, low, close}

def aggregate_bar(trades, current_bar):
    """Update OHLC bar with new tick"""
    price = trades.get("now")
    if price is None:
        return current_bar
    price = float(price)
    if current_bar["open"] is None:
        current_bar["open"] = price
        current_bar["high"] = price
        current_bar["low"] = price
    current_bar["high"] = max(current_bar["high"], price)
    current_bar["low"] = min(current_bar["low"], price)
    current_bar["close"] = price
    return current_bar

def save_bar(bar):
    df = pd.DataFrame([bar])
    header = not os.path.exists(BAR_FILE)
    df.to_csv(BAR_FILE, mode="a", index=False, header=header)

def save_prediction(pred_df):
    header = not os.path.exists(PRED_FILE)
    pred_df.to_csv(PRED_FILE, mode="a", index=False, header=header)

def get_prediction():
    df = pd.DataFrame(bars[-LOOKBACK:])  # last LOOKBACK bars
    x_df = df[["open", "high", "low", "close"]]
    x_timestamp = df["timestamp"]

    # generate y timestamps (future 12 bars, each 5 min)
    last_ts = df["timestamp"].iloc[-1]
    y_timestamp = [last_ts + timedelta(minutes=5*(i+1)) for i in range(PRED_LEN)]

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    print("=== Prediction for next 1h ===")
    print(pred_df)

    save_prediction(pred_df)
    return pred_df

def main():
    global bars
    current_bar = {"timestamp": None, "open": None, "high": None, "low": None, "close": None}
    next_bar_time = None

    while True:
        try:
            data = q.real([STOCK_CODE])
            stock = data.get(STOCK_CODE, {})
            now = datetime.now()

            # Initialize bar timing
            if next_bar_time is None:
                # round up to next 5 min
                minute = (now.minute // 5 + 1) * 5
                if minute == 60:  # handle rollover
                    next_bar_time = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
                else:
                    next_bar_time = now.replace(minute=minute, second=0, microsecond=0)
                current_bar["timestamp"] = now.replace(second=0, microsecond=0)

            # Update bar
            current_bar = aggregate_bar(stock, current_bar)

            # If bar finished
            if now >= next_bar_time:
                bars.append(current_bar.copy())
                print("✅ New 5-min bar:", current_bar)

                # Save bar
                save_bar(current_bar)

                # reset bar
                current_bar = {"timestamp": now.replace(second=0, microsecond=0), "open": None, "high": None, "low": None, "close": None}
                next_bar_time = next_bar_time + timedelta(minutes=5)

                # run prediction if enough bars
                if len(bars) >= LOOKBACK:
                    get_prediction()

            time.sleep(1)

        except Exception as e:
            print("❌ Error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()

