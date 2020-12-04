#!/usr/bin/env python
"""
  The code serves only for educational purposes so it
  is not optimized in view of speed or numerical calculations.
  For the real problems you should always use algorithms.
  from known,  well tested libraries, wherever possible.

  Author: Wojtek Krzemien
  Date: 17.06 2018
"""

def sortXAndY(x,y):
  perm = x.argsort()
  return x[perm],y[perm] 
