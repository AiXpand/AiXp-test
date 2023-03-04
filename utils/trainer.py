# -*- coding: utf-8 -*-
"""
Copyright (C) 2017-2021 Andrei Damian, andrei.damian@me.com,  All rights reserved.

This software and its associated documentation are the exclusive property of the creator. 
Unauthorized use, copying, or distribution of this software, or any portion thereof, 
is strictly prohibited.

Parts of this software are licensed and used in software developed by Knowledge Investment Group SRL.
Any software proprietary to Knowledge Investment Group SRL is covered by Romanian and  Foreign Patents, 
patents in process, and are protected by trade secret or copyright law.

Dissemination of this information or reproduction of this material is strictly forbidden unless prior 
written permission from the author.

"""

import numpy as np
import torch as th
from copy import deepcopy
from time import time


from models.simple_classifiers import AbstractClassifier

def train_classifier(model : AbstractClassifier, 
                     dct_result,
                     train_loader, test_loader,
                     optimizer=th.optim.NAdam, lr=1e-4,
                     loss=th.nn.CrossEntropyLoss(), epochs=100,
                     earlystopping=True, patience=5,
                     log_func=print, progress_delta=4e-4,
                     freeze_after=None,
                     ):
  log_func("Starting training of '{}'".format(model.__class__.__name__))
  
  optim = optimizer(params=model.parameters(), lr=lr)
  lost_patience = 0
  not_freezed = True
  freeze_check_cnt = 0
  
  
  for epoch in range(1, epochs + 1):
    t0epoch = time()
    n_batches = 0
    for th_bx, th_by in train_loader:
      n_batches += 1
      t0batch = time()
      th_yh = model(th_bx)
      th_loss = loss(input=th_yh, target=th_by)
      optim.zero_grad()
      th_loss.backward()
      optim.step()
      dct_result['BATCH_TIMINGS'].append(time() - t0batch)
    # end epoch
    dct_result['EPOCH_TIMINGS'].append(time() - t0epoch)
    
    if (epoch % 2) == 0:
      t0dev = time()
      dev_result = model.evaluate(test_loader, use_inference=False)
      dct_result['EVAL_TIMINGS'].append(time() - t0dev)
    else:
      t0dev = time()
      dev_result = model.evaluate(test_loader, use_inference=True)
      dct_result['INFER_TIMINGS'].append(time() - t0dev)
    
    if (dct_result['BEST_DEV'] + progress_delta) < dev_result:
      dct_result['BEST_DEV'] = dev_result
      dct_result['BEST_EPOCH'] = epoch
      best_weights = deepcopy(model.state_dict())
      lost_patience = 0
      print("*", end='', flush=True)
    else:
      print(".", end='', flush=True)
      lost_patience += 1
    
    if freeze_after is not None and epoch >= freeze_after and not_freezed:
      frozen_weights = deepcopy(model.freeze_backbone())
      not_frozen = deepcopy(model.readout.state_dict())
      k1 = list(frozen_weights.keys())[0]
      k2 = list(not_frozen.keys())[0]
      th_test_f = frozen_weights[k1]
      th_test_nf = not_frozen[k2]
      not_freezed = False
      
    if freeze_after is not None and epoch in [freeze_after + 2, freeze_after + 4]:
      th_test_new_f = model.backbone.state_dict()[k1]
      th_test_new_nf = model.readout.state_dict()[k2]     
      freeze_check_cnt += 1
      if th.allclose(th_test_f, th_test_new_f, atol=1e-3):
        log_func("Freeze sanity #{} check ok.".format(freeze_check_cnt))
      else:
        log_func("Freeze sanity #{} check NOT ok.".format(freeze_check_cnt))
      freeze_check_cnt += 1
      if not th.allclose(th_test_nf, th_test_new_nf, atol=1e-3):
        log_func("Freeze sanity #{} check ok.".format(freeze_check_cnt))
      else:
        log_func("Freeze sanity #{} check NOT ok.".format(freeze_check_cnt))
      
      
      
    if lost_patience > patience:
      model.load_state_dict(best_weights)
      dct_result['MODEL'] = model
      break
  # end epochs
  
  if freeze_after is not None:
    th_test_new_f = model.backbone.state_dict()[k1]
    th_test_new_nf = model.readout.state_dict()[k2]
    freeze_check_cnt += 1
    if th.allclose(th_test_f, th_test_new_f, atol=1e-3):
      log_func("Freeze sanity #{} check ok.".format(freeze_check_cnt))
    else:
      log_func("Freeze sanity #{} check NOT ok.".format(freeze_check_cnt))
    freeze_check_cnt += 1
    if not th.allclose(th_test_nf, th_test_new_nf, atol=1e-3):
      log_func("Freeze sanity #{} check ok.".format(freeze_check_cnt))
    else:
      log_func("Freeze sanity #{} check NOT ok.".format(freeze_check_cnt))

  fn = model.save_to_file()
  dct_result['FN'] = fn
  dct_result['NR_BATCHES'] = n_batches

  dct_result['EPOCH_MIN']     = np.min(dct_result['EPOCH_TIMINGS']).round(5)
  dct_result['EPOCH_MAX']     = np.max(dct_result['EPOCH_TIMINGS']).round(5)
  dct_result['EPOCH_STD']     = np.std(dct_result['EPOCH_TIMINGS']).round(5)

  dct_result['BATCH_MIN']     = np.min(dct_result['BATCH_TIMINGS']).round(5)
  dct_result['BATCH_MAX']     = np.max(dct_result['BATCH_TIMINGS']).round(5)
  dct_result['BATCH_STD']     = np.std(dct_result['BATCH_TIMINGS']).round(5)

  dct_result['EVAL_MIN']      = np.min(dct_result['EVAL_TIMINGS']).round(5)
  dct_result['EVAL_MAX']      = np.max(dct_result['EVAL_TIMINGS']).round(5)
  dct_result['EVAL_STD']      = np.std(dct_result['EVAL_TIMINGS']).round(5)

  dct_result['INFER_MIN']      = np.min(dct_result['INFER_TIMINGS']).round(5)
  dct_result['INFER_MAX']      = np.max(dct_result['INFER_TIMINGS']).round(5)
  dct_result['INFER_STD']      = np.std(dct_result['INFER_TIMINGS']).round(5)
  
  return dct_result
    
  