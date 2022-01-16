# implementando camada LSTM
import keras
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D
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

def modelomlp(data, num_classes, model_config):
  n_steps = data.shape[1]
  model = Sequential()
  model.add(Dense(model_config.neumlp, activation=model_config.funcactiv, input_dim=n_steps))
  if model_config.use_drop:
    model.add(Dropout(model_config.drop))
  model.add(Dense(num_classes, activation=model_config.funcout))
  opt = Adam(lr=model_config.opt_lr,beta_1=model_config.opt_beta)
  model.compile(optimizer=opt, loss=model_config.loss, metrics=[model_config.metrics])
  return model

def modelocnn(data, num_classes, model_config):
  model = Sequential()
  input_shape=(data.shape[1:])
  model.add(Conv2D(model_config.neucnn_1, kernel_size=model_config.kernel_size,padding = 'same', activation=model_config.funcactiv, input_shape=input_shape))
  if model_config.use_drop:
    model.add(Dropout(model_config.drop))
  if model_config.neucnn_2 != 0:
    model.add(Conv2D(model_config.neucnn_2, kernel_size=model_config.kernel_size,padding = 'same',activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))
  model.add(Flatten())
  if model_config.neumlp_1 != 0:
    model.add(Dense(model_config.neumlp_1, activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))  
  if model_config.neumlp_2 != 0:
    model.add(Dense(model_config.neumlp_2, activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))
  model.add(Dense(num_classes, activation=model_config.funcout))
  model.compile(optimizer=model_config.optimizer, loss=model_config.loss, metrics=[model_config.metrics])
  return model
