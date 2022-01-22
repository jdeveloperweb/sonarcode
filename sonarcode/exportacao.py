import os
import pickle
import pandas as pd
import pandas
import matplotlib.pyplot as plt

class Caminho:
  def __init__(self):
    self.logs = "/gdrive/MyDrive/LPS/GOLTZ/resultados/logs/"
    self.modelos =  "/gdrive/MyDrive/LPS/GOLTZ/resultados/modelos/"
    self.curvas = "/gdrive/MyDrive/LPS/GOLTZ/resultados/curvas/"
    self.history = "/gdrive/MyDrive/LPS/GOLTZ/resultados/history/"
    self.his_obj = "/gdrive/MyDrive/LPS/GOLTZ/resultados/history/objeto/"
    self.his_plan = "/gdrive/MyDrive/LPS/GOLTZ/resultados/history/planilha/"
  
  def diretorio(self, caminho):
    if os.path.isdir(caminho):
      None
    else:
      os.mkdir(caminho)
    return caminho

class Analise:
  def __init__(self):
    path_init = Caminho()
    self.objeto = path_init.his_obj
    self.planilha = path_init.his_plan

  def save_var(self, historico,_end, nome, test, fold): # salvar objeto do modelo treinado
    with open(self.objeto+'/'+_end+'/'+str(nome) + '_test_' + str(test) + '_fold_' + str(fold), 'wb') as file:
        pickle.dump(historico.history, file)
  
  def save_csv(self, data,_end, nome, test, fold): # salvar planilha do modelo treinado
    with open(self.planilha+'/'+_end+'/'+str(nome) + '_test_' + str(test) + '_fold_' + str(fold) + '.csv', 'wb') as file:
      data.to_csv(file.name, encoding='utf-8', index=True)


class Resultados:
  def __init__(self, metrica):
    path_init = Caminho()
    self.objeto = path_config.his_obj
    self.planilha = path_config.his_plan
    self.curvas = path_config.curvas
    self.historico = path_config.history
    self.dir_files = [x for x in os.listdir(self.planilha) if x.startswith(metrica)]

  def csv_individual(self, ntest, nfold):
    teste = "test_" + str(ntest)
    fold = "fold_" + str(nfold)
    files = [x for x in self.dir_files if teste in x]
    data = [x for x in files if fold in x]
    return pandas.read_csv(self.planilha + str(data[0])), data[0]
  
  def csv_lista(self, num_list, teste=True):
    result = []
    if teste:
      num_list = "test_" + str(num_list)
    else:
      num_list = "fold_" + str(num_list)
    files = [x for x in self.dir_files if num_list in x]
    [result.append(pandas.read_csv(self.planilha + str(x))) for x in files]
    return result, files

  def plot_accloss(self, data, nome):
    plt.plot(data.accuracy)
    plt.plot(data.val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(str(self.curvas) + str(nome) + '_acc.png')  
    plt.close()
    plt.plot(data.loss) # summarize history for loss
    plt.plot(data.val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(str(self.curvas) + str(nome) + '_loss.png')
    plt.close()
