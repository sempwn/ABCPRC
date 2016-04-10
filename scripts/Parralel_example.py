# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:20:04 2016

@author: Mike Irvine (sempwn)
@email: michael.irvine.mai@gmail.com
"""
import sys
sys.path.append('./../ABCPRC/')
import ABCPRC as prc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# initial test
runSims = True
if runSims:
    plt.hist(prc.xs,bins=30)
    p = prc.Parameters(prc.tols,prc.vs,prc.xs,prc.sim)
    res,accepted_dists = prc.abcprcParralel(p,N=100)
p1,p2 = res[0],res[1]

x1 = pd.Series(p1[0,:], name="k")
x2 = pd.Series(p2[0,:], name="m")
plt.figure()
g = sns.jointplot(x=x1, y=x2,kind="scatter")#,xlim=(0,0.2),ylim=(0,21))
g.ax_joint.plot(prc.p0_true,prc.p1_true,'ro')
#plt.savefig('Example_posterior_e1.pdf',bbox_inches='tight')

x1 = pd.Series(p1[5,:], name="k")
x2 = pd.Series(p2[5,:], name="m")
plt.figure()
g = sns.jointplot(x=x1, y=x2,kind="scatter")#,xlim=(0,0.2),ylim=(0,21))
g.ax_joint.plot(prc.p0_true,prc.p1_true,'ro')
#plt.savefig('Example_posterior_e6.pdf',bbox_inches='tight')

x1 = pd.Series(p1[-1,:], name="k")
x2 = pd.Series(p2[-1,:], name="m")
plt.figure()
g = sns.jointplot(x=x1, y=x2,kind="scatter")#,xlim=(0,0.2),ylim=(0,21))
g.ax_joint.plot(prc.p0_true,prc.p1_true,'ro')