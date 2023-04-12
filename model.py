import csv
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from matplotlib import pyplot as plt

data = np.loadtxt("dataset-normalized.csv", delimiter=";")

# input data is everything but class
X = data[:, :-1]
# output data is only class
Y = data[:, -1]

# determine how many neurons to use in hidden layer
HL = {"I": 5, "IO2": int((5 + 18) / 2), "IO": 5 + 18}
hidden_neurons = "IO2"

five_fold = KFold(n_splits=5, shuffle=True)
errors = []

for i, (train, test) in enumerate(five_fold.split(X)):
    model = Sequential()
    model.add(Dense(HL[hidden_neurons], activation="relu", input_dim=18))
    model.add(Dense(5, activation="softmax"))

    def rmse(y_true, y_pred):
        return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))

    keras.optimizers.SGD(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['mse', 'accuracy'])

    history = model.fit(X[train], to_categorical(Y[train]), epochs=500, batch_size=500, verbose=False)

    scores = model.evaluate(X[test], to_categorical(Y[test]), verbose=False)
    errors.append(scores)
    print(f"Fold {i}: {errors[-1]}")

    fig, ax = plt.subplots(3)

    ax[0].set_title("CE loss")
    ax[0].plot(history.history["loss"], label="train")
    #ax[0].plot(history.history["val_loss"], label="test")

    ax[1].set_title("MSE")
    ax[1].plot(history.history["mse"], label="train")
    #ax[1].plot(history.history["val_mse"], label="test")

    ax[2].set_title("Accuracy")
    ax[2].plot(history.history["accuracy"], label="train")
    #ax[2].plot(history.history["val_accuracy"], label="test")

    fig.suptitle(f"{HL[hidden_neurons]} neurons in hidden layer - fold {i+1}")
    plt.savefig(f"img/history-{HL[hidden_neurons]}-{i+1}.png")



print(f"Total (using {HL[hidden_neurons]} neurons in hidden layer):")
print(f"Mean CE loss: {np.mean([x[0] for x in errors])}")
print(f"Mean MSE: {np.mean([x[1] for x in errors])}")
print(f"Mean accuracy: {np.mean([x[2] for x in errors])}")