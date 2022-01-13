import numpy as np 

def generate_data_trgt_pair(processed_data_dict, trgt_label_map=None):

    if trgt_label_map is None:
        trgt_label_map = classe_target
    
    trgt = np.concatenate(
        [trgt_label_map[cls_name]*np.ones(Sxx[0].shape[0]) 
        for cls_name, run in processed_data_dict.items() 
        for run_name, Sxx in run.items()]
        )
    
    data = np.concatenate(
        [Sxx[0]
        for cls_name, run in processed_data_dict.items() 
        for run_name, Sxx in run.items()], axis=0
        )
    return data, trgt
  
