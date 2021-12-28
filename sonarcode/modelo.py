# implementando camada LSTM
import keras
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.layers import Conv1D
from tensorflow.keras.optimizers import Adam


def modelolstm(n_timesteps, n_features, n_outputs, model_config):
  model = Sequential()
  model.add(LSTM(model_config.neulstm, input_shape=(n_timesteps,n_features)))
  if model_config.use_drop:
    model.add(Dropout(model_config.drop))
  model.add(Flatten())
  # if model_config.neumlp_1.isalnum():
  if model_config.neumlp_1 != 0:
    model.add(Dense(model_config.neumlp_1, activation=model_config.funcactiv))
  model.add(Dense(n_outputs, activation=model_config.funcout))
  model.compile(optimizer= model_config.optimizer, loss= model_config.loss, metrics=[model_config.metrics])
  return model