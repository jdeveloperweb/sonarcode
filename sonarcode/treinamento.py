import numpy as np
from sklearn import preprocessing
import pandas as pd

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
