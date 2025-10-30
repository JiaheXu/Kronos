import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
import numpy as np

def plot_prediction(gts, preds):
    # for pred in preds:
    #     print(pred.shape)
    gts = np.stack(gts)
    # print("gts: ",gts.shape)
    preds = np.stack(preds)

    for i in range(gts.shape[1]):
        gt_flat = gts[:,i].squeeze()
        result_flat = preds[:,i].squeeze()

        # X-axis: time or index
        x = np.arange(len(gt_flat))

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(x, gt_flat, label="Ground Truth", linewidth=2)
        plt.plot(x, result_flat, label="Prediction", linewidth=2, linestyle='--')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Ground Truth vs Prediction {} mins ahead".format(i*5+5))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the figure
        plt.savefig("{}_mins.png".format(i*5+5), dpi=300)
        # plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. Prepare Data
df = pd.read_csv("./data/sz002877.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 240
pred_len = 12
start = 720 - lookback - pred_len - 120

gts = []
results = []
close_errors = []

for i in range( start, 720 - lookback - pred_len):
# for i in range( start, start + 10):
    # 4. Make Prediction

    x_df = df.loc[ i : i+lookback - 1, ['open', 'high', 'low', 'close']]
    x_timestamp = df.loc[ i : i+lookback - 1 ,  'timestamps']

    y_timestamp = df.loc[ i + lookback : i + lookback + pred_len - 1, 'timestamps']

    # print('y: ', y_timestamp)
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    result = pred_df.tail(1)
    # results.append( result )


    gt = df.loc[ i + lookback : i + lookback + pred_len - 1, ['open', 'high', 'low', 'close']]
    gt_timestamp = df.loc[  i + lookback : i + lookback + pred_len - 1,  'timestamps']
    gt.index = gt_timestamp.values
    gt.index.name = "timestamps"
    
    # Align on common columns
    common_cols = gt.columns.intersection(pred_df.columns)

    # Calculate difference (gt - pred_df) for overlapping timestamps
    close_diff = abs( gt["close"] - pred_df["close"] )
    # print("close_diff: ", close_diff.shape)
    close_errors.append(close_diff)
    # print('gt["close"]: ', gt["close"])
    # print('pred_df["close"]: ', pred_df["close"])
    gt_np = gt["close"].to_numpy()
    result_np = pred_df["close"].to_numpy()
    
    # print("gt_np: ", gt_np.shape)
    # print("result_np: ", result_np.shape)

    gts.append( gt_np )
    results.append( result_np )


close_errors = np.array(close_errors)
# print("close_errors: ",close_errors.shape)
mse = np.mean(close_errors, axis = 0)
mxe = np.max(close_errors, axis = 0)


print('mse: ', mse)
print('mxe: ', mxe)

x = np.arange(1, len(mse)+1)  # e.g., [1, 2, ..., 12]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, mse, marker='o', label="MSE", color='blue')
plt.plot(x, mxe, marker='s', label="Max Error", color='red')
plt.title("MSE and Maximum Error over 12 Prediction Steps")
plt.xlabel("Prediction Step")
plt.ylabel("Error")
plt.xticks(x)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("error.png", dpi=300)

plot_prediction( gts, results)