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

import traceback
import platform
from time import sleep


class Tester:
  def __init__(self):
    return
  
  def P(self, s):
    print(str(s), flush=True)
    return
  
  
  def gpu_info(self, show=False, mb=False, current_pid=False, debug=False):
    """
    Collects GPU info. Must have torch installed & non-mandatory nvidia-smi

    Parameters
    ----------
    show : bool, optional
      show data as gathered. The default is False.
    mb : bool, optional
      collect memory in MB otherwise in GB. The default is False.
    current_pid: bool, optional
      return data only for GPUs used by current process or all if current process does
    not use GPU
    

    Returns
    -------
    lst_inf : list of dicts
      all GPUs info from CUDA:0 to CUDA:n.

    """

    def _main_func():
      try:
        # first get name
        import torch as th
        import os
      except:
        self.P("ERROR: `gpu_info` call failed - PyTorch probably is not installed:\n{}".format(
          traceback.format_exc())
        )
        return None
      
      self.P("1:__no_gpu_avail:" + str(vars(self).get('__no_gpu_avail', False)))
      
      if vars(self).get('__no_gpu_avail', False):
        self.P("Return as already no gpu has been discovered")
        return None

      nvsmires = None
      try:
        from pynvml.smi import nvidia_smi
        import pynvml
        nvsmi = nvidia_smi.getInstance()
        self.P("nvsmi:" + str(nvsmi))
        nvsmires = nvsmi.DeviceQuery('memory.free, memory.total, memory.used, utilization.gpu, temperature.gpu')
        self.P("nvsmires:" + str(nvsmires))
        pynvml_avail = True
        if len(nvsmires) == 0:
          self.P("Setting __no_gpu_avail=True")
          vars(self)['__no_gpu_avail'] = True
        else:
          vars(self)['__no_gpu_avail'] = False
      except:
        pynvml_avail = False
        self.P("Error in pynvml:\n{}".format(traceback.format_exc()))
        vars(self)['__no_gpu_avail'] = True
      
      self.P("2:__no_gpu_avail:" + str(vars(self).get('__no_gpu_avail', False)))

      lst_inf = []
      # now we iterate all devices
      n_gpus = th.cuda.device_count()
      if n_gpus > 0:
        th.cuda.empty_cache()
      current_pid_has_usage = False
      current_pid_gpus = []

      try:
        for device_id in range(n_gpus):
          dct_device = {}
          device_props = th.cuda.get_device_properties(device_id)
          dct_device['NAME'] = device_props.name
          dct_device['TOTAL_MEM'] = round(
            device_props.total_memory / 1024 ** (2 if mb else 3),
            2
          )
          mem_total = None
          mem_allocated = None
          gpu_used = None
          gpu_temp = None
          gpu_temp_max = None
          if pynvml_avail and nvsmires is not None and 'gpu' in nvsmires:
            dct_gpu = nvsmires['gpu'][device_id]
            mem_total = round(
              dct_gpu['fb_memory_usage']['total'] / (1 if mb else 1024),
              2
            )  # already from th
            mem_allocated = round(
              dct_gpu['fb_memory_usage']['used'] / (1 if mb else 1024),
              2
            )
            gpu_used = dct_gpu['utilization']['gpu_util']
            if isinstance(gpu_used, str):
              gpu_used = -1
            gpu_temp = dct_gpu['temperature']['gpu_temp']
            gpu_temp_max = dct_gpu['temperature']['gpu_temp_max_threshold']

            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            processes = []
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
              dct_proc_info = {k.upper(): v for k,v in proc.__dict__.items()}
              used_mem = dct_proc_info.pop('USEDGPUMEMORY', None)
              dct_proc_info['ALLOCATED_MEM'] = round(
                used_mem / 1024 ** (2 if mb else 3) if used_mem is not None else 0.0,
                2
              )
              processes.append(dct_proc_info)
              if dct_proc_info['PID'] == os.getpid():
                current_pid_has_usage = True
                current_pid_gpus.append(device_id)
            #endfor
            dct_device['PROCESSES'] = processes
            dct_device['USED_BY_PROCESS'] = device_id in current_pid_gpus
          else:
            str_os = platform.platform()
            ## check if platform is Tegra and record
            if 'tegra' in str_os.lower():
              # we just record the overall fre memory
              mem_total = self.get_machine_memory()
              mem_allocated = mem_total  - self.get_avail_memory()
              gpu_used = 1
              gpu_temp = 1
              gpu_temp_max = 100
              if not self._done_first_smi_error and nvsmires is not None:
                self.P("Running `gpu_info` on Tegra platform: {}".format(nvsmires), color='r')
                self._done_first_smi_error = True
            elif not self._done_first_smi_error:
              str_log = "ERROR: Please make sure you have both pytorch and pynvml in order to monitor the GPU"
              str_log += "\nError info: pynvml_avail={}, nvsmires={}".format(pynvml_avail, nvsmires)
              self.P(str_log)
              self._done_first_smi_error = True
          #endif
          dct_device['ALLOCATED_MEM'] = mem_allocated
          dct_device['FREE_MEM'] = -1
          if all(x is not None for x in [mem_total, mem_allocated]):
            dct_device['FREE_MEM'] = round(mem_total - mem_allocated,2)
          dct_device['MEM_UNIT'] = 'MB' if mb else 'GB'
          dct_device['GPU_USED'] = gpu_used
          dct_device['GPU_TEMP'] = gpu_temp
          dct_device['GPU_TEMP_MAX'] = gpu_temp_max

          lst_inf.append(dct_device)
        #end for all devices
      except Exception as e:
        self.P("gpu_info exception for device_id {}:\n{}".format(device_id, e), color='r')

      if show:
        self.P("GPU information for {} device(s):".format(len(lst_inf)), color='y')
        for dct_gpu in lst_inf:
          for k, v in dct_gpu.items():
            self.P("  {:<14} {}".format(k + ':', v), color='y')

      if current_pid and current_pid_has_usage:
        return [lst_inf[x] for x in current_pid_gpus]
      else:
        return lst_inf
    #enddef

    # if multiple threads call at the same time the method `log.gpu_info`, then `pynvml_avail` will be False
    #   (most probably due to the queries that are performed using nvidia_smi)
    res = _main_func()
    return res


if __name__ == '__main__':
  eng = Tester()
  
  res = eng.gpu_info(debug=False)
  eng.P("Warm 1")
  eng.P("Warm 2")
  eng.P("Warm 3")
  eng.P("Testing...")
  res = eng.gpu_info(debug=True)
  eng.P("  Done")
  print(res)
  
  