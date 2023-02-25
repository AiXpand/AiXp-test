import pkg_resources

import sys
import traceback

from core.lib import cwd, P
  
from time import sleep
if __name__ == '__main__':
  packs = [x for x in pkg_resources.working_set]
  maxlen = max([len(x.key) for x in packs]) + 1
  packs = [
    "{}{}".format(x.key + ' ' * (maxlen - len(x.key)), x.version) for x in packs
  ]
  packs = sorted(packs)
  path = cwd()
  P("Hello world startup in '{}', py: {}...".format(path,sys.version))
  P("Packages: \n{}".format("\n".join(packs)))
  try:
    import torch as th
    P('th: {}'.format(th.__version__))
    err = False
    t = th.tensor([1,2,3] * 1000)
    if th.cuda.is_available():
      dev = th.device('cuda')
      P('cuda: {}'.format(th.cuda.get_device_name('cuda')))
      t = t.to(dev)
      P('{}:{}'.format(t, t.device))
    else:
      P("Cuda is not available...")
         
  except:
    P("{}".format(traceback.format_exc()))
    err = True
        
  for step in range(1, 30):        
    P('Hello world step {}'.format(step))
    sleep(1)
      