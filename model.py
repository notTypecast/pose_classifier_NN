import csv
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import GlorotNormal
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from random import randint

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

# determine how many neurons to use in hidden layer
HL = {"O": 5, "IO2": int((5 + 18) / 2), "IO": 5 + 18}
hidden_neurons = "IO" # VAR: change to determine first hidden layer neurons
hidden_neurons_2 = "IO2" # VAR: change to determine second hidden layer neurons
hidden_neurons_3 = "O" # VAR: change to determine third hidden layer neurons (ignored if hidden_neurons_2 is None)

# other hyperparameters
lr = 0.001 # VAR: learning rate
m = 0.2 # VAR: momentum constant
r = 0 # VAR: regularization constant

# determine whether to use early stopping using validation set
early_stopping = False # VAR: determines whether early stopping is enabled
restore_best = True # VAR: determines whether to restore best weights without early-stopping (early_stopping must be False)

if early_stopping or restore_best:
    data = np.loadtxt("dataset-train.csv", delimiter=";")
    val_data = np.loadtxt("dataset-test.csv", delimiter=";")

    # input data is everything but class
    X = data[:, :-1]
    # output data is only class
    Y = data[:, -1]

    # same for validation data
    valX = val_data[:, :-1]
    valY = to_categorical(val_data[:, -1])
else:
    data = np.loadtxt("dataset-normalized.csv", delimiter=";")

    X = data[:, :-1]
    Y = data[:, -1]

five_fold = KFold(n_splits=5, shuffle=True)
metrics = []
histories = []

# train model using 5-fold CV
for i, (train, test) in enumerate(five_fold.split(X)):
    initializer = GlorotNormal(seed=randint(0, 100000))

    model = Sequential()
    model.add(Dense(HL[hidden_neurons], activation="relu", input_dim=18, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(r), bias_regularizer=l2(r)))

    if hidden_neurons_2 is not None:
        model.add(Dense(HL[hidden_neurons_2], activation="relu", kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(r), bias_regularizer=l2(r)))
        if hidden_neurons_3 is not None:
            model.add(Dense(HL[hidden_neurons_3], activation="relu", kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=l2(r), bias_regularizer=l2(r)))

    model.add(Dense(5, activation="softmax", kernel_initializer=initializer, bias_initializer=initializer))

    keras.optimizers.SGD(learning_rate=lr, momentum=m)
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['mse', 'accuracy'])

    if r:
        cb = [EarlyStopping(monitor="accuracy", min_delta=0.0001, patience=5, start_from_epoch=15, restore_best_weights=True)]
    elif early_stopping:    
        cb = [EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=8, start_from_epoch=15, restore_best_weights=True)]
    else:
        cb = [EarlyStopping(monitor="val_loss", patience=1, start_from_epoch=90, restore_best_weights=True)]

    histories.append(model.fit(X[train], to_categorical(Y[train]), validation_data=(valX, valY) if early_stopping or restore_best else (), epochs=100, batch_size=100, callbacks=cb if early_stopping or r or restore_best else []))

    scores = model.evaluate(X[test], to_categorical(Y[test]), verbose=False)
    metrics.append(scores)
    print(f"Fold {i}: {metrics[-1]}")

fig, ax = plt.subplots(3)

ax[0].set_title("CE loss")
ax[0].plot(get_avg_history(histories, "loss"), label="train")
if early_stopping or restore_best:
    ax[0].plot(get_avg_history(histories, "val_loss"), label="validation")
    ax[0].legend()

ax[1].set_title("MSE")
ax[1].plot(get_avg_history(histories, "mse"), label="train")
if early_stopping or restore_best:
    ax[1].plot(get_avg_history(histories, "val_mse"), label="validation")
    ax[1].legend()

ax[2].set_title("Accuracy")
ax[2].plot(get_avg_history(histories, "accuracy"), label="train")
if early_stopping or restore_best:
    ax[2].plot(get_avg_history(histories, "val_accuracy"), label="validation")
    ax[2].legend()

fig.suptitle(f"{HL[hidden_neurons]} neurons in hidden layer - n: {lr}, m: {m}, r: {r}")
plt.savefig(f"img/history-{HL[hidden_neurons]}{f'-{HL[hidden_neurons_2]}' if hidden_neurons_2 is not None else ''}{f'-{HL[hidden_neurons_3]}' if hidden_neurons_3 is not None else ''}{f'-lr_{lr}-m_{m}' if m else ''}{f'-r_{r}' if r else ''}{'-es' if early_stopping else ('-rb' if restore_best else '')}.png")

# show results
print(f"Total (using {HL[hidden_neurons]} neurons in hidden layer):")
print(f"Mean CE loss: {np.mean([x[0] for x in metrics])}")
print(f"Mean MSE: {np.mean([x[1] for x in metrics])}")
print(f"Mean accuracy: {np.mean([x[2] for x in metrics])}")