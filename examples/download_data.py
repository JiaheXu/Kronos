import akshare as ak
import pandas as pd
import os


def download_stock_5min_kline(symbol: str, start_date: str = "2025-09-01", period: str = "5", adjust: str = "qfq"):
    """
    Download 5-minute K-line data for a given A-share stock symbol and save to CSV.

    Parameters:
        symbol (str): Stock symbol in format like 'sz002277' or 'sh600519'
        start_date (str): Date from which to start filtering data (format: YYYY-MM-DD)
        period (str): Minute interval ('1', '5', '15', '30', '60')
        adjust (str): Price adjustment method ('qfq' for 前复权, etc.)
    """

    # Fetch data
    try:
        df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust=adjust)
    except Exception as e:
        print(f"❌ Failed to fetch data for {symbol}: {e}")
        return

    # Convert day column to datetime
    if "day" not in df.columns:
        print(f"❌ 'day' column not found in data for {symbol}")
        return

    df["day"] = pd.to_datetime(df["day"])
    df = df[df["day"] >= pd.to_datetime(start_date)]

    # Rename columns
    df = df.rename(columns={
        "day": "timestamps",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "vol"
    })

    # Add missing 'amt' column if not present
    if "amount" in df.columns:
        df["amt"] = df["amount"]
    else:
        df["amt"] = float("nan")

    # Set datetime index and select columns
    df.set_index("timestamps", inplace=True)
    df = df[["open", "high", "low", "close", "vol", "amt"]]

    # Create output directory
    #output_dir = os.path.join(symbol, "csv")
    #os.makedirs(output_dir, exist_ok=True)

    # Output file path
    output_file = os.path.join(f"./data/{symbol}.csv")

    # Save to CSV
    df.to_csv(output_file, index_label="timestamps")

    print(f"✅ Data saved to: {output_file}")
    print(f"📊 Total rows: {len(df)}")


if __name__ == "__main__":
    download_stock_5min_kline("sz002277")
    download_stock_5min_kline("sz002877")
