from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, Adamax, Nadam, Adadelta, Adagrad, SGD
from sklearn.model_selection import train_test_split
from keras import regularizers
import optuna
import tensorflow as tf
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
import pandas as pd
from keras.backend import clear_session

DATA_FILE = "normalized_15_game_average_final.csv"
clear_session()
data = pd.read_csv(DATA_FILE)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
x, y = data.iloc[:, :-1], data.iloc[:, -1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.9)

model = Sequential()
model.add(
    Dense(
        19,
        activation='relu',
        input_shape=(x_train.shape[1],),
        use_bias=True,
        name="Input",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    )
)
model.add(Dense(1, activation="sigmoid"))
lr = 0.005033330002355446
opt = None
optimizer = "Adagrad"
if optimizer == "RMSprop":
    opt = RMSprop(lr=lr)
elif optimizer == "Adam":
    opt = Adam(lr=lr)
elif optimizer == "Adamax":
    opt = Adamax(lr=lr)
elif optimizer == "Nadam":
    opt = Nadam(lr=lr)
elif optimizer == "Adadelta":
    opt = Adadelta(lr=lr)
elif optimizer == "Adagrad":
    opt = Adagrad(lr=lr)
elif optimizer == "SGD":
    opt = SGD(lr=lr)


model.compile(
    loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]
)
model.fit(x_train, y_train, batch_size=1, epochs=10000, verbose=0, validation_data=(x_valid, y_valid),
          callbacks=[earlystopping])

# Evaluate the model accuracy on the validation set.
score_train = model.evaluate(x_train, y_train, verbose=0)
score = model.evaluate(x_valid, y_valid, verbose=0)
score_test = model.evaluate(x_test, y_test, verbose=0)
print("train: {0} {1}".format(float(score_train[0]), float(score_train[1])))
print("valid: {0} {1}".format(float(score[0]), float(score[1])))
print("test: {0} {1}".format(float(score_test[0]), float(score_test[1])))
