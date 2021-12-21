# implementando camada LSTM
import keras
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.layers import Conv1D
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing
import pandas as pd


def modelolstm(n_timesteps, n_features, n_outputs, model_config):
  model = Sequential()
  model.add(LSTM(model_config.neulstm, input_shape=(n_timesteps,n_features)))
  if model_config.use_drop:
    model.add(Dropout(model_config.drop))
  model.add(Flatten())
  model.add(Dense(n_outputs, activation=model_config.funcout))
  model.compile(optimizer= model_config.optimizer, loss= model_config.loss, metrics=[model_config.metrics])
  # model.add(Dropout(model_config.drop))
  # model.add(Flatten())
  # model.add(Dense(model_config.neumlp, activation=model_config.funcactiv))
  return model

# função de padronização
def padronizaconjunto(data, tipo):
  #cria o objeto obj_std
  if tipo == 'mapstd':
    obj_std = preprocessing.StandardScaler().fit(data)
  elif tipo == 'mapstd_rob':
    obj_std = preprocessing.RobustScaler().fit(data)  
  elif tipo == 'mapminmax':
    obj_std = preprocessing.MinMaxScaler().fit(data)
  data_std = obj_std.transform(data) #aplica o padronizador nos dados
  return data_std, obj_std


# função para formatar os dados de entrada para a rede cnn
def format_inputlstm(data):
  data_array = np.array(data[:])
  x_out = data_array.reshape(data_array.shape[0],data_array.shape[1],1)
  return x_out
