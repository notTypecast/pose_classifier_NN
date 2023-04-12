import csv
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from matplotlib import pyplot as plt

def get_avg_history(histories, metric):
    """
    Arguments:
        histories: list of keras history objects
        metric: metric to get average history for
    Returns list of average history for all history objects included, for given metric
    All history objects must have given metric
    History objects do not need to be of the same length; average value of all existing histories for time t is returned
    """
    avg_history = []
    max_history = max(histories, key=lambda h:len(h.history[metric]))

    for i, _ in enumerate(max_history.history[metric]):
        all_curr_vals = []
        for history in histories:
            try:
                all_curr_vals.append(history.history[metric][i])
            except IndexError:
                pass

        avg_history.append(np.mean(all_curr_vals))

    return avg_history

class MinValAccuracyEarlyStopping(EarlyStopping):
    def __init__(self, min_accuracy=None, **kwargs):
        super().__init__(**kwargs)
        self.min_accuracy = min_accuracy
        self.allow_early_stopping = min_accuracy is None

    def on_epoch_end(self, epoch, logs=None):
        if self.min_accuracy is not None:
            accuracy = logs.get("val_accuracy")
            if accuracy is not None and accuracy >= self.min_accuracy:
                self.allow_early_stopping = True

            if not self.allow_early_stopping:
                return

        return super().on_epoch_end(epoch, logs)

data = np.loadtxt("dataset-train.csv", delimiter=";")
val_data = np.loadtxt("dataset-test.csv", delimiter=";")

# input data is everything but class
X = data[:, :-1]
# output data is only class
Y = data[:, -1]

# same for validation data
valX = val_data[:, :-1]
valY = to_categorical(val_data[:, -1])

# determine how many neurons to use in hidden layer
HL = {"I": 5, "IO2": int((5 + 18) / 2), "IO": 5 + 18}
hidden_neurons = "IO"

# determine whether to use early stopping using validation set
early_stopping = False

five_fold = KFold(n_splits=5, shuffle=True)
metrics = []
histories = []

# train model using 5-fold CV
for i, (train, test) in enumerate(five_fold.split(X)):
    model = Sequential()
    model.add(Dense(HL[hidden_neurons], activation="relu", input_dim=18))
    model.add(Dense(5, activation="softmax"))

    def rmse(y_true, y_pred):
        return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))

    keras.optimizers.SGD(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['mse', 'accuracy'])

    cb = [MinValAccuracyEarlyStopping(min_accuracy=0.6, monitor="val_loss", min_delta=0.001, patience=4, restore_best_weights=True)]

    histories.append(model.fit(X[train], to_categorical(Y[train]), validation_data=(valX, valY) if early_stopping else (), epochs=100, batch_size=100, callbacks=cb if early_stopping else []))

    scores = model.evaluate(X[test], to_categorical(Y[test]), verbose=False)
    metrics.append(scores)
    print(f"Fold {i}: {metrics[-1]}")

fig, ax = plt.subplots(3)

ax[0].set_title("CE loss")
ax[0].plot(get_avg_history(histories, "loss"), label="train")
if early_stopping:
    ax[0].plot(get_avg_history(histories, "val_loss"), label="validation")
    ax[0].legend()

ax[1].set_title("MSE")
ax[1].plot(get_avg_history(histories, "mse"), label="train")
if early_stopping:
    ax[1].plot(get_avg_history(histories, "val_mse"), label="validation")
    ax[1].legend()

ax[2].set_title("Accuracy")
ax[2].plot(get_avg_history(histories, "accuracy"), label="train")
if early_stopping:
    ax[2].plot(get_avg_history(histories, "val_accuracy"), label="validation")
    ax[2].legend()

fig.suptitle(f"{HL[hidden_neurons]} neurons in hidden layer")
plt.savefig(f"img/history-{HL[hidden_neurons]}{'-es' if early_stopping else ''}.png")

# show results
print(f"Total (using {HL[hidden_neurons]} neurons in hidden layer):")
print(f"Mean CE loss: {np.mean([x[0] for x in metrics])}")
print(f"Mean MSE: {np.mean([x[1] for x in metrics])}")
print(f"Mean accuracy: {np.mean([x[2] for x in metrics])}")