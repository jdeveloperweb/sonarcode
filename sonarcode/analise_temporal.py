import numpy as np
from scipy.signal import decimate 
from scipy.signal import cheby2, cheb2ord
from scipy import signal

def time_process(data, ftype_input, zero_phase_input, process_config):
  dec = process_config.decimate
  if dec == 1: # decimação do sinal
    data_dec = data.copy()
  else:
    data_dec = decimate(data,dec, ftype=ftype_input, zero_phase=zero_phase_input)
  h = dec_filtro(process_config) # filtro passa baixa
  data_fil = signal.sosfilt(h, data_dec)
  if process_config.normalize == True:
    data_norm = (data_fil-data_fil.min())/(data_fil.max()-data_fil.min()) # normalização
  else:
    data_norm = data_fil
  return data_norm

# projetar o filtro
def dec_filtro(process_config):
  dec = process_config.decimate
  Fs = int(process_config.fs/dec) 
  fp = int(process_config.fp/dec)  
  fs = int(process_config.fc/dec)
  Ap = process_config.Ap 
  As = process_config.As
  wp = fp/(Fs/2)  
  ws = fs/(Fs/2)  
  N, wc = signal.cheb2ord(wp, ws, Ap, As)
  filtro = signal.cheby2(N, As, wc, 'low', output='sos')
  return filtro
