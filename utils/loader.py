# -*- coding: utf-8 -*-
"""
Copyright 2017-2022 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


* NOTICE:  
*   All information contained herein is, and remains the property of Knowledge Investment Group SRL.  
*   The intellectual and technical concepts contained herein are proprietary to Knowledge Investment Group SRL
*   and may be covered by Romanian and Foreign Patents, patents in process, and are protected by trade secret 
*   or copyright law. Dissemination of this information or reproduction of this material * is strictly forbidden 
*   unless prior written permission is obtained from Knowledge Investment Group SRL.



@copyright: Lummetry.AI
@author: Lummetry.AI - AID
@project: 
@description:
Created on Sat Feb 25 10:36:19 2023
"""

import torch as th
import torchvision as tv



def get_data(device=None, batch_size=128):
      
  trn_ds = tv.datasets.MNIST(root='./data',download=True,train=True)  
  th_trn_x = trn_ds.data.to(device).reshape(-1,1,28,28)
  th_trn_x = th_trn_x / 255.
  th_trn_y = trn_ds.targets.to(device)
  train_bytes = th_trn_x.nelement() * th_trn_y.element_size()
  print("Train: {:.2f} GB".format( train_bytes / (1024**3)), flush=True)
  
  
  tst_ds = tv.datasets.MNIST(root='./data',download=True,train=False)  
  th_tst_x = tst_ds.data.to(device).reshape(-1,1,28,28)
  th_tst_x = th_tst_x / 255.
  th_tst_y = tst_ds.targets.to(device)
  
  train_dl = th.utils.data.DataLoader(
    th.utils.data.TensorDataset(th_trn_x, th_trn_y),
    batch_size=batch_size,
    shuffle=True,
  )  
  test_dl = th.utils.data.DataLoader(
    th.utils.data.TensorDataset(th_tst_x, th_tst_y),
    batch_size=batch_size,
    shuffle=True,
  )  
  return train_dl, test_dl
  
  