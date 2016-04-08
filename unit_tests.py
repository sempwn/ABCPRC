# -*- coding: utf-8 -*-
"""
Unit tests for ABCPRC module
Created on Fri Apr  1 16:08:31 2016

@author: u1472179
"""


import IBMSolver as ibm
from nose import with_setup
from nose.tools import *
import ABCPRC as abc

def setup():
    pass

def tear_down():
    pass

@with_setup(setup,tear_down)
def test_data_import():
    m = abc.ABC()
    assert (m.parameters.xs.size > 0)

@with_setup(setup)
@raises(NameError)
def test_run_before_trace():
    m = abc.ABC()
    m.trace()





