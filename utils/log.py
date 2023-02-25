
import os
import pkg_resources
import json

from collections import OrderedDict
from time import time, strftime, localtime

logdata = OrderedDict({})
FMT = '%Y-%m-%d %H:%M:%S'
FILE_FMT = '%Y%m%d_%H%M%S'
FILE_PREFIX =  strftime(FILE_FMT, localtime(time()))


def time_to_str(tm):
  return strftime(FMT, localtime(tm))
  
def LOG(s):
  now = time()
  str_now = time_to_str(now)
  idx = len(logdata) + 1
  logdata[str_now] = str(s)
  print('[{:04} {}] {}'.format(idx,str_now, str(s)), flush=True)
  save_json(logdata, folder='logs')
  return
  

def cwd():
  return os.getcwd()
  
  
def get_packages():
  packs = [x for x in pkg_resources.working_set]
  maxlen = max([len(x.key) for x in packs]) + 1
  packs = [
    "{}{}".format(x.key + ' ' * (maxlen - len(x.key)), x.version) for x in packs
  ]
  packs = sorted(packs)  
  return packs


def save_json(obj, fn='', folder='results'):
  running_in_docker = os.environ.get('AIXP_DOCKER', False)
  subfolder = os.path.join('./output', folder)
  os.makedirs(subfolder, exist_ok=True)
  fn = "{}{}{}.txt".format(
    FILE_PREFIX, 
    "_dokr" if running_in_docker else '_bare',
    ('_' + fn) if len(fn) > 0 else ''
  )  
  path = os.path.join(subfolder, fn)
  with open(path, 'wt') as fh:
    json.dump(obj, fh, indent=4)
  return
  