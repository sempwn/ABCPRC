# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:54:06 2016

@author: u1472179
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#import sys
#sys.path.append('./../')
import ABCPRC
if __name__ == '__main__':
    # initial test
    runSims = True
    if runSims:
        plt.hist(ABCPRC.xs,bins=30)
        #res = ABCPRC.ABCPRC()
    p1,p2 = res['p1'],res['p2']
    
    x1 = pd.Series(p1[1,:], name="k")
    x2 = pd.Series(p2[1,:], name="m")
    plt.figure()
    g = sns.jointplot(x=x1, y=x2,kind="scatter",xlim=(0,0.2),ylim=(0,21))
    g.ax_joint.plot(0.05,10,'ro')
    #plt.savefig('Example_posterior_e1.pdf',bbox_inches='tight')
    
    x1 = pd.Series(p1[5,:], name="k")
    x2 = pd.Series(p2[5,:], name="m")
    plt.figure()
    g = sns.jointplot(x=x1, y=x2,kind="scatter",xlim=(0,0.2),ylim=(0,21))
    g.ax_joint.plot(0.05,10,'ro')
    #plt.savefig('Example_posterior_e6.pdf',bbox_inches='tight')
    
    x1 = pd.Series(p1[10,:], name="k")
    x2 = pd.Series(p2[10,:], name="m")
    plt.figure()
    g = sns.jointplot(x=x1, y=x2,kind="scatter",xlim=(0,0.2),ylim=(0,21))
    g.ax_joint.plot(0.05,10,'ro')