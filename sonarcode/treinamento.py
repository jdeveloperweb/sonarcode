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
  data_std = pd.DataFrame(data_std) # transforma em dataFrame
  return data_std, obj_std


# função para formatar os dados de entrada para a rede cnn
def format_inputlstm(data):
  data_array = np.array(data[:])
  x_out = data_array.reshape(data_array.shape[0],data_array.shape[1],1)
  return x_out


# Difinindo quem é o melhor modelo
def melhor_modelo(sp_fold,val_fold):  
  n_max = sp_fold.count(max(sp_fold))
  if n_max > 1:
    ind_model = [(n) for n, x in enumerate(sp_fold) if x== max(sp_fold)] # Quais indices?
    val_model = [val_fold[x] for x in ind_model] # Valores de validação dos indices
    best_model = val_model.index(max(val_model))
  else:
    best_model = sp_fold.index(max(sp_fold))
    return best_model
  
  
  
