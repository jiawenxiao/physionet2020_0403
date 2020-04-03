#!/usr/bin/env python

import numpy as np

def get_12ECG_features(data,header_data):
    
    data_num=np.zeros((1,12,7500))

    length=data.shape[1]
    if length>=7500:
        data_num[:,:,:]=data[:,:7500]
    else:
        data_num[:,:,:length]=data
    
    data_num=data_num.transpose([0,2,1]) 
    
    data_external=np.zeros((1,3))
    
    for lines in header_data:
        if lines.startswith('#Age'):
            age=lines.split(': ')[1].strip()
            if age=='NaN':
                age='60'     
        if lines.startswith('#Sex'):
            sex=lines.split(': ')[1].strip()
            
            
    length=data.shape[1]
    data_external[:,0]=float(age)/100
    data_external[:,1]=np.array(sex=='Male').astype(int) 
    data_external[:,2]=length/30000   
    
    return data_num,data_external