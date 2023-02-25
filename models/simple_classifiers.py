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
Created on Sat Feb 25 10:20:21 2023
"""
import os
import numpy as np
import torch as th

def get_conv_out(w, k, p, s):
  return int((w - k + 2 * p) / s + 1)

class AbstractClassifier(th.nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    return

  def batch_predict(self, inputs, return_th=False):
    dev = self.readout.weight.device
    if isinstance(inputs, np.ndarray):
      th_x = th.tensor(inputs, device=dev)
    elif isinstance(inputs, th.Tensor):
      th_x = inputs
    else:
      raise ValueError("Unknown inputs of type {}".format(type(inputs)))
    if th_x.device != dev:
      th_x = th_x.to(dev)
    with th.no_grad():
      self.eval()
      th_yh = self(th_x)
      self.train()
    if not return_th:
      np_yh = th_yh.cpu().numpy()
      return np_yh
    else:
      return th_yh
    
    
  def load_from_file(self, fn='simple_mnist_clf.th', directory='./output/models'):
    os.makedirs(directory, exist_ok=True)
    fn = os.path.join(directory, fn)
    state_dict = th.load(fn)
    self.load_state_dict(state_dict)
    return
  
  def save_to_file(self, fn='simple_mnist_clf.th', directory='./output/models'):
    os.makedirs(directory, exist_ok=True)
    fn = os.path.join(directory, fn)
    state_dict = self.state_dict()
    th.save(state_dict, fn)
    return fn
  
  def evaluate(self, test_loader):
    assert isinstance(test_loader, th.utils.data.DataLoader)
    res = []
    for th_bx, th_by in test_loader:
      th_yh = self.batch_predict(th_bx, return_th=True)
      th_yp = th_yh.argmax(axis=1).reshape(-1)
      th_res = th_yp == th_by
      res.append(th_res)
    # end for
    th_all_res = th.concat(res)
    np_res = th_all_res.cpu().numpy()
    result = np_res.sum() / np_res.shape[0]
    return result
  
class ConvBlock(th.nn.Module):
  def __init__(self, input_f, f, k, s, bn, pad, input_w, act=th.nn.ReLU6):
    super().__init__()
    self.conv = th.nn.Conv2d(input_f, f, kernel_size=k, padding=pad, stride=s)
    if bn:
      self.bn = th.nn.BatchNorm2d(f)
    else:
      self.bn = None
    self.act = act()
    self.output_width = get_conv_out(w=input_w, k=k, p=pad, s=s)
    self.out_f = f
    return
  
  def forward(self, inputs):
    th_x = self.conv(inputs)
    if self.bn is not None:
      th_x = self.bn(th_x)
    return self.act(th_x)
  
  def __repr__(self):
    s = super().__repr__()
    s += " => Output {}".format((self.out_f, self.output_width, self.output_width))
    return s
    
    

class SimpleConvBackbone(th.nn.Module):
  def __init__(self, input_shape=[1,28,28], convs=[8, 16, 32], k=3, bn=True, stride=2, pad=0, **kwargs):
    super().__init__()
    prev_f = input_shape[0]
    prev_w = input_shape[1]
    _convs = []
    for i,conv in enumerate(convs):
      assert prev_w > 2, "Volume after {} conv is only {} and cannot support a new conv k={}. Check your proposed architecture".format(i + 1, (prev_f, prev_w, prev_w), k)
      _convs.append(ConvBlock(
        input_f=prev_f,
        f=conv,
        k=k,
        s=stride,
        bn=bn,
        input_w=prev_w,
        pad=pad
      ))
      prev_f = conv
      prev_w = _convs[-1].output_width

    self.convs = th.nn.ModuleList(_convs)
    self.final_conv_size = prev_w
    self.final_nunits = prev_f * (prev_w * prev_w)
    return
  
  def forward(self, inputs):
    th_x = inputs
    for cnv in self.convs:
      th_x = cnv(th_x)
    return th_x

class SimpleFC(th.nn.Module):
  def __init__(self, input_size=None, fcs=[64], dropout=0.5, **kwargs):
    super().__init__()
    assert input_size is not None
    prev_u = input_size
    _fcs = []
    if dropout > 0:
      _fcs.append(th.nn.Dropout(dropout))
    for fc in fcs:
      _fcs.append(th.nn.Linear(prev_u, fc))
      _fcs.append(th.nn.ReLU6())
      if dropout > 0:
        _fcs.append(th.nn.Dropout(dropout))
      prev_u = fc
    
    self.fcs = th.nn.ModuleList(_fcs)
    self.final_nunits = prev_u
    return
 
  def forward(self, inputs):
    th_x = inputs
    for fc in self.fcs:
      th_x = fc(th_x)
    return th_x    
  
  

class SimpleMNISTClassifier(AbstractClassifier):
  def __init__(self, **kwargs):
    super().__init__()
    self.backbone = SimpleConvBackbone(input_shape=(1,28,28), **kwargs)
    self.conv_to_fc = th.nn.Flatten()
    self.fc = SimpleFC(input_size=self.backbone.final_nunits, **kwargs)
    self.readout = th.nn.Linear(self.fc.final_nunits, 10)    
    return
    
  def forward(self, inputs):
    th_x = inputs
    th_x = self.backbone(th_x)
    th_x = self.conv_to_fc(th_x)
    th_x = self.fc(th_x)
    th_out = self.readout(th_x)
    return th_out
  




if __name__ == '__main__':
  
  from copy import deepcopy
  
  def compare_state(s1,s2):
    
    result = True
    if list(s1.keys()) != list(s2.keys()):
      result = False
    else:
      for k in s1:
        if (s1[k] != s2[k]).sum() > 0:
          result = False
    return result
  
  m = SimpleMNISTClassifier(convs=[8, 16, 32])
  
  inp = th.rand((32,1,28,28))
  out1 = m.batch_predict(inp, return_th=True)
  
  m.save_to_file()
  
  m2 = SimpleMNISTClassifier(convs=[8, 16, 32]).to(th.device('cuda'))
  s2 = m2.state_dict()
  s2c = deepcopy(m2.state_dict())
  m2.load_from_file()
  print(compare_state(s2, m2.state_dict()))
  print(compare_state(s2c, m2.state_dict()))
  
  out2 = m2.batch_predict(inp, return_th=True)
  
  assert th.allclose(out1, out2.cpu())
  
  print(m)
  