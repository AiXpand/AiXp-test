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
import sys
import os
import json
import platform
import multiprocessing as mp

from time import time

from models.simple_classifiers import SimpleMNISTClassifier

from utils.log import cwd, LOG, get_packages, save_json, history_report, get_empty_train_log, show_train_log
from utils.loader import get_data
from utils.trainer import train_classifier


__VER__ = '0.7.1'

def runs_with_debugger():
  gettrace = getattr(sys, 'gettrace', None)
  if gettrace is None:
    return False
  else:
    return not gettrace() is None

def test_main():
  running_in_docker = os.environ.get('AIXP_DOCKER', False) != False
  ee_id = os.environ.get('EE_ID', 'bare_app')
  show_packs = os.environ.get('SHOW_PACKS')
  BATCH_SIZE = 512
  EPOCHS = 100
  EARLY_STOPPING, PATIENCE = True, 5
  FORCE_CPU = False
  
  in_debug = runs_with_debugger()
  LOG("Running in DEBUG mode" if in_debug else "Running in normal mode")
  
  if len(sys.argv) > 1 and 'cpu' in sys.argv[1].lower():
    FORCE_CPU = True 
  packs = get_packages()
  path = cwd()
  LOG("Running {} test v{} '{}', py: {}, OS: {}, Docker: {}".format(
    ee_id,
    __VER__,
    path, sys.version.split(' ')[0], platform.platform(), running_in_docker
    ), mark=True
  )
  
  LOG("Show packages: {}".format(show_packs))
  if show_packs in ['Yes', 'YES', 'yes']:
    LOG("Packages: \n{}".format("\n".join(packs)))
  t_start = time()
  
  dev = th.device('cpu')
  
  str_os = platform.platform().lower()
  if "wsl" in str_os or "microsoft" in str_os or "windows" in str_os:
    host = "windows"
  elif 'ubuntu' in str_os:
    host = 'linux'
  else:
    host = str_os[:5]  
  
  device_name = "cpu {} core {}".format(mp.cpu_count(), host)
  
  if FORCE_CPU:
    LOG("Forcing default use of CPU")
  else:
    if th.cuda.is_available():
      dev = th.device('cuda')
      device_name = th.cuda.get_device_name('cuda')
      LOG('GPU device available: {}'.format(device_name))
    else:
      LOG("Cuda is not available...")
  
  LOG("Loading train and test data...")
  train_loader, test_loader = get_data(dev, batch_size=BATCH_SIZE)
  LOG("  Done data loading.")
  
  LOG("Creating a model...")
  model = SimpleMNISTClassifier(
    convs=[32, 64, 128],
    fcs=[256],
    pad=1,
    bn=True,
  )
  model.to(dev)
  _dev = next(model.parameters()).device
  LOG("  Following model created and deployed on device {}:\n{}".format(_dev, model))
  
  LOG("Training the model on {} : '{}'...".format(_dev,device_name), mark=True)
  dct_result = get_empty_train_log()
  dct_result['TEST_VERSION'] = __VER__
  dct_result['DOCKER'] = running_in_docker
  dct_result['DEVICE'] = device_name
  dct_result['OS_SHORT'] = host
  _ = train_classifier(
    model=model, 
    train_loader=train_loader, 
    test_loader=test_loader,
    dct_result=dct_result,
    optimizer=th.optim.NAdam, 
    lr=1e-4,
    loss=th.nn.CrossEntropyLoss(), 
    epochs=EPOCHS,
    earlystopping=EARLY_STOPPING, 
    patience=PATIENCE,    
    log_func=LOG    ,
    freeze_after=2 if in_debug else None,
  )
  LOG("  Done training.")
  total_time = round(time() - t_start,2)
  LOG("v{} results after {:.2f}s process:".format(__VER__, total_time), mark=True)
  dct_result['TOTAL_TIME'] = total_time
  dct_result['INFER_GAIN'] = round(
    (np.mean(dct_result['EVAL_TIMINGS']) - np.mean(dct_result['INFER_TIMINGS'])) / 
    np.mean(dct_result['EVAL_TIMINGS']) * 100,1
  )
  dct_result['INFER_CNT'] = len(dct_result['EVAL_TIMINGS'])
  show_train_log(dct_result)
  save_json(dct_result, fn=str(dev.type))
  history_report()
  LOG("Send this to the team:\n\n{}".format(json.dumps(dct_result, indent=4)), mark=True)
  return
  
if __name__ == '__main__':
  test_main()