# -*- coding: utf-8 -*-
"""
Unit tests for ABCPRC module
Created on Fri Apr  1 16:08:31 2016

@author: u1472179
"""



from nose import with_setup
from nose.tools import *
import ABCPRC as prc

def setup():
    pass

def tear_down():
    pass


def test_load_package():
    import ABCPRC

@with_setup(setup)
def test_data_import():
    m = prc.ABC()
    assert (m.parameters.xs.size > 0)

@with_setup(setup)
@raises(NameError)
def test_run_before_trace():
    m = prc.ABC()
    m.trace()

@raises(NameError)
def test_run_before_paramMAP():
    m = prc.ABC()
    m.paramMAP()

@raises(NameError)
def test_run_before_fitSummary():
    m = prc.ABC()
    m.fitSummary()





