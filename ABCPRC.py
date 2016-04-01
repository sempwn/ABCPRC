"""
Implementation of ABC-PRC code 
reference: Parameter inference in small world network disease models with approximate Bayesian Computational methods
Walker et al. Physica A 2010 
Created on Thu March 7 14:33:33 2016

@author: Michael Irvine (Sempwn)
"""
import sys, os
import traceback
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool
from joblib import Parallel, delayed 
import multiprocessing
from functools import partial
import IBMSolver as ibm

class KeyboardInterruptError(Exception): pass
    
#start parralel pool
def start_process():
    print 'Starting', multiprocessing.current_process().name
  
       
#default parameters
num_cores = multiprocessing.cpu_count()
tols = np.array([1000,500,250,190,150,125,100,75,50,45,40,35,30])/1000.
N = 200
params = 2
v1 = stats.gamma(1.0).rvs
v2 = stats.gamma(100.0).rvs
p0_true = 1.0
p1_true = 100.0
vs = [v1,v2] #prior distribution random variable functions.
rw_var = 0.01

#generate some fake data
##
# Test bimodal model.
#ps[0] - peak of first Gaussian
#ps[1] - peak of second Gaussian
def biModSim(*ps):
    xs = np.zeros(1000)
    for i in range(xs.size):
        r = np.random.rand()
        if r < 0.5:
            xs[i] = stats.norm.rvs(loc=ps[0])
        else:
            xs[i] = stats.norm.rvs(loc=ps[1])
    return xs
## 
# Test negative binomial model.
# ps[0] - mean of nbinom distribution
# ps[1] - aggregation k factor.
def nbinSim(*ps):
    p = ps[0]/(ps[1]+ps[0])
    if p == 0:
        return np.zeros(1000)
    else:
        return stats.nbinom(n=ps[0],p=p).rvs(size=1000)
    
#define Sim function:
fakeSim = nbinSim#ibm.mfOutSim#
#input data.
xs = fakeSim(p0_true,p1_true,0)


def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()



##
# Main function to implement ABC-PRC scheme.
# input - tols : tolerances, N : no. of particles.
def ABCPRCParralel(tols=tols,N=N):
    #pool = Pool(processes=4) 
    #set-up matrices to record particles
    # structure is pRecs[i][j,k] i= parameter, j = time, k = particle.
    pRecs = []
    pRecs.append( np.zeros((tols.size,N)) )
    pRecs.append( np.zeros((tols.size,N)) )



    for t, tol in enumerate(tols):

        if (t==0):
            #initialise first particles from the priors v1,v2
            pRecs[0][0,:] = vs[0](size=N)
            pRecs[1][0,:] = vs[1](size=N)
        else:
            func = partial(particlesF,t,pRecs[0],pRecs[1])
            try:
                #vfunc = np.vectorize(func)                
                #res = pool.map(func, range(N))#vfunc(range(N))#
                res = Parallel(n_jobs=num_cores)(delayed(func)(i) for i in range(N))
            except KeyboardInterrupt:
                print 'got ^C while pool mapping, terminating the pool'
       
            except Exception, e:
                print 'got exception: %r' % (e,)

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                traceback.print_exc()
                
            pRecs[0][t,:] = np.array([a[0] for a in res]).flatten()
            pRecs[1][t,:] = np.array([a[1] for a in res]).flatten()
            '''          
            a_star = np.zeros(params)
            for i in range(N):
                #TODO: implenet parralel bit here
                print 't = {}. i = {}'.format(t,i)
                #random sample from previous alpha with perturbation
                rho = tols[t]+1.
                while(rho>tols[t]):
                    r = np.random.randint(0,high=N)
                    a_star[0] = stats.gamma.rvs(p1A[t-1,r]/rw_var,scale=rw_var)#p1A[t-1,r] + stats.norm.rvs(scale=0.1) 
                    a_star[1] = stats.gamma.rvs(p2A[t-1,r]/rw_var,scale=rw_var)#p2A[t-1,r] + stats.norm.rvs(scale=0.1)
                                      
                    rho = distFunc(a_star[0],a_star[1])
                p1A[t,i] = a_star[0]
                p2A[t,i] = a_star[1]
            '''
        update_progress(float(t)/len(tols))

    #pool.close(); pool.join()
    res = pRecs
    update_progress(10.)
    sys.stdout.write("\n")
    return res
                
def ABCPRC(tols=tols,N=N):
    pool = Pool(processes=4) 
    #set-up matrices to record particles
    p1A = np.zeros((tols.size,N))
    p2A = np.zeros((tols.size,N))
    
    
    
    for t, tol in enumerate(tols):
    
        if (t==0):
            #initialise first particles from the priors vs
            p1A[0,:] = vs[0](size=N)
            p2A[0,:] = vs[1](size=N)
        else:
            #func = partial(particlesF, t,p1A,p2A)
            
            #vfunc = np.vectorize(func)                
            #res = vfunc(range(N))#map(func, range(N))
        
            #e = sys.exc_info()[0]
            #pool.close(); pool.join()
            #print 'Errored so closed pool.'
            #print e
            
            #p1A[t,:] = np.array([a for a,b in res]).flatten()
            #p2A[t,:] = np.array([b for a,b in res]).flatten()
                  
            a_star = np.zeros(params)
            for i in range(N):
                #TODO: implenet parralel bit here
                print 't = {}. i = {}'.format(t,i)
                #random sample from previous alpha with perturbation
                rho = tols[t]+1.
                while(rho>tols[t]):
                    r = np.random.randint(0,high=N)
                    a_star[0] = stats.gamma.rvs(p1A[t-1,r]/rw_var,scale=rw_var)#p1A[t-1,r] + stats.norm.rvs(scale=0.1) 
                    a_star[1] = stats.gamma.rvs(p2A[t-1,r]/rw_var,scale=rw_var)#p2A[t-1,r] + stats.norm.rvs(scale=0.1)
                                      
                    rho = distFunc(a_star[0],a_star[1],0)
                p1A[t,i] = a_star[0]
                p2A[t,i] = a_star[1]
            
        update_progress(float(t)/len(tols))
    
    pool.close(); pool.join()
    res = {'p1': p1A, 'p2' : p2A}
    update_progress(10.)
    sys.stdout.write("\n")
    return res   

   
##
# Calculate distance between two empirical distributions.
# input parameters for model.
# output: distance between generated data and data xs.
def distFunc(p1,p2,ii):
    ys = fakeSim(p1,p2,ii)
    if (np.sum(ys)==0):
        return np.inf
    else:
        kernely = stats.gaussian_kde(ys)
        kernelx = stats.gaussian_kde(xs)
        xx = np.linspace(np.min(xs),np.max(xs)) #range over data.
        return stats.entropy(kernelx(xx),qk=kernely(xx)) #KL-divergence.

##
# Filter particles step.
def particlesF(t,p1A,p2A,ii):
    np.random.seed()
    sys.stdout.flush()
    a_star = np.zeros(2)
    rho = tols[t]+1.

    while(rho>tols[t]):
        try:
            r = np.random.randint(0,high=N)
            a_star[0] = stats.gamma.rvs(p1A[t-1,r]/rw_var,scale=rw_var)#p1A[t-1,r] + stats.norm.rvs(scale=0.1) 
            a_star[1] = stats.gamma.rvs(p2A[t-1,r]/rw_var,scale=rw_var)#p2A[t-1,r] + stats.norm.rvs(scale=0.1)
                              
            rho = distFunc(a_star[0],a_star[1],ii)
        
        except KeyboardInterrupt:
            raise KeyboardInterruptError()
    p1 = a_star[0]
    p2 = a_star[1] 
    return p1,p2


    
    
if __name__ == '__main__':
    # initial test
    runSims = True
    if runSims:
        plt.hist(xs,bins=30)
        res = ABCPRCParralel()
    p1,p2 = res[0],res[1]
    
    x1 = pd.Series(p1[1,:], name="k")
    x2 = pd.Series(p2[1,:], name="m")
    plt.figure()
    g = sns.jointplot(x=x1, y=x2,kind="scatter")#,xlim=(0,0.2),ylim=(0,21))
    g.ax_joint.plot(p0_true,p1_true,'ro')
    #plt.savefig('Example_posterior_e1.pdf',bbox_inches='tight')
    
    x1 = pd.Series(p1[5,:], name="k")
    x2 = pd.Series(p2[5,:], name="m")
    plt.figure()
    g = sns.jointplot(x=x1, y=x2,kind="scatter")#,xlim=(0,0.2),ylim=(0,21))
    g.ax_joint.plot(p0_true,p1_true,'ro')
    #plt.savefig('Example_posterior_e6.pdf',bbox_inches='tight')
    
    x1 = pd.Series(p1[-1,:], name="k")
    x2 = pd.Series(p2[-1,:], name="m")
    plt.figure()
    g = sns.jointplot(x=x1, y=x2,kind="scatter")#,xlim=(0,0.2),ylim=(0,21))
    g.ax_joint.plot(p0_true,p1_true,'ro')
    #plt.savefig('Example_posterior_e11.pdf',bbox_inches='tight')
    
    #plt.plot(p1[10,:],p2[10,:],'bo',alpha=0.2)
    #plt.savefig('Example_posterior.pdf',bbox_inches='tight')
    
    '''
    runSims = False
    if runSims:
        ks = np.linspace(0.01,0.1,num=10)
        ms = np.linspace(1.0,100.0,num=10) 
        k_res,m_res = np.zeros((10,10)),np.zeros((10,10))
        for i,k in enumerate(ks):
            for j,m in enumerate(ms):
                    print 'At point {},{}'.format(i,j)
                    xs = fakeSim(k,m)
                    v1 = stats.gamma(k).rvs
                    v2 = stats.gamma(m).rvs
                    res = ABCPRC()
                    p1,p2 = res['p1'],res['p2']
                    k_res[i,j] = 100*(np.mean(p1[10,:]) - k)/k
                    m_res[i,j] = 100* (np.mean(p2[10,:]) - m)/m
    #plt.imshow(abs(k_res))
    #plt.colorbar()
    ax = sns.heatmap(abs(m_res),xticklabels=ks,yticklabels=ms,cbar=False); plt.xlabel('k'); plt.ylabel('m')
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_label('Relative error in m (%)')
    plt.savefig('relative_m_error.pdf')
    
    plt.figure()
    #plt.imshow(abs(m_res))
    ax = sns.heatmap(abs(k_res),xticklabels=ks,yticklabels=ms,cbar=False); plt.xlabel('k'); plt.ylabel('m')
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_label('Relative error in k (%)')
    plt.savefig('relative_k_error.pdf')
    '''