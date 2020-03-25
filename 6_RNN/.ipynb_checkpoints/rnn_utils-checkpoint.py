import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow.keras.losses import mean_squared_error


# Function that generates batch_size time series, each of lenght n_steps
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time-offset1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time-offset2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)

# plot a time-series with max 1 forecast point
def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", n_steps=50):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])
    
# plot a time-series with unlimited forecast lenght
def plot_multiple_forecasts(X, Y, Y_pred, seq_id=0):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[seq_id, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[seq_id, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[seq_id, :, 0], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)


def last_time_step_mse(y_true, y_pred):
    return mean_squared_error(y_true[:, -1], y_pred[:, -1])

# plot learning curves from history object
def plot_learning_curves(loss, val_loss, axis = [1, 20, 0, 0.05]):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis(axis)
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)