import numpy as np
import torch as th
import sys
from time import time

from models.simple_classifiers import SimpleMNISTClassifier

from utils.log import cwd, LOG, get_packages, save_json
from utils.loader import get_data
from utils.trainer import train_classifier


if __name__ == '__main__':
  BATCH_SIZE = 512
  EPOCHS = 100
  EARLY_STOPPING, PATIENCE = True, 5
  FORCE_CPU = False
  
  if len(sys.argv) > 1 and 'cpu' in sys.argv[1].lower():
    FORCE_CPU = True 
  
  packs = get_packages()
  path = cwd()
  LOG("Running test '{}', py: {}...".format(path,sys.version))
  LOG("Packages: \n{}".format("\n".join(packs)))
  t_start = time()
  dev = th.device('cpu')
  if FORCE_CPU:
    LOG("Forcing default use of CPU")
  else:
    if th.cuda.is_available():
      dev = th.device('cuda')
      LOG('GPU device available: {}'.format(th.cuda.get_device_name('cuda')))
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
  LOG("  Following model created and deployed on device {}:\n{}".format(next(model.parameters()).device, model))
  
  LOG("Training the model...")
  res = train_classifier(
    model, 
    train_loader, 
    test_loader,
    optimizer=th.optim.NAdam, 
    lr=1e-4,
    loss=th.nn.CrossEntropyLoss(), 
    epochs=EPOCHS,
    earlystopping=EARLY_STOPPING, 
    patience=PATIENCE,    
    log_func=LOG    
  )
  LOG("  Done training.")
  total_time = round(time() - t_start,2)
  LOG("Results after {:.2f}s process:".format(total_time))
  m = res.pop('MODEL')
  res['TOTAL_TIME'] = total_time
  res['MODEL'] = m
  maxk = max(len(x) for x in res)
  for k in res:
    if isinstance(res[k], list):
      fv = round(np.mean(res[k]),5)
      v = "{:.4f}s".format(fv)    
      res[k] = fv
    else:
      v = str(res[k])     
      if not isinstance(res[k], (int, float)):
        res[k] = v
    LOG("{} {}".format(
      k + ' ' * (maxk - len(k)), 
      v,
    ))
  save_json(res, fn=str(dev.type))
  