#!/usr/bin/env python
"""
  The code serves only for educational purposes so it
  is not optimized in view of speed or numerical calculations.
  For the real problems you should always use algorithms.
  from known,  well tested libraries, wherever possible.

  Author: Wojtek Krzemien
  Date: 17.06 2018
"""
import numpy as np

def true_f(x):
  return np.sin(2*np.pi*x)

def sampleWithNoise(nSamples = 100, noiseFraction = 0.3, f = true_f):
  x = np.sort(np.random.rand(nSamples))
  return x, f(x) + noiseFraction* np.random.normal(0,1,len(x))

