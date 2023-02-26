import torch as th
import sys
import os
import json
from time import time

from models.simple_classifiers import SimpleMNISTClassifier

from utils.log import cwd, LOG, get_packages, save_json, history_report, get_empty_train_log, show_train_log
from utils.loader import get_data
from utils.trainer import train_classifier


__VER__ = '0.2.2.0'

def test_main():
  running_in_docker = os.environ.get('AIXP_DOCKER', False) != False
  ee_id = os.environ.get('EE_ID', 'bare_app')
  BATCH_SIZE = 512
  EPOCHS = 100
  EARLY_STOPPING, PATIENCE = True, 5
  FORCE_CPU = False
  
  if len(sys.argv) > 1 and 'cpu' in sys.argv[1].lower():
    FORCE_CPU = True 
  
  packs = get_packages()
  path = cwd()
  LOG("Running {} test v{} '{}', py: {}, Docker container: {}...".format(
    ee_id,
    __VER__,
    path,sys.version, running_in_docker
  ))
  LOG("Packages: \n{}".format("\n".join(packs)))
  t_start = time()
  dev = th.device('cpu')
  device_name = 'cpu'
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
  LOG("  Following model created and deployed on device {}:\n{}".format(next(model.parameters()).device, model))
  
  LOG("Training the model...")
  dct_result = get_empty_train_log()
  dct_result['DOCKER'] = running_in_docker
  dct_result['DEVICE'] = device_name
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
    log_func=LOG    
  )
  LOG("  Done training.")
  total_time = round(time() - t_start,2)
  LOG("Results after {:.2f}s process:".format(total_time))
  dct_result['TOTAL_TIME'] = total_time
  show_train_log(dct_result)
  save_json(dct_result, fn=str(dev.type))
  history_report()
  LOG("Send this to AiXp team:\n\n{}".format(json.dumps(dct_result, indent=4)), mark=True)
  return
  
if __name__ == '__main__':
  test_main()