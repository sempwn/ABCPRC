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
p0_true = v1()
p1_true = v2()
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
def nbinSim(*ps):
    '''
    # Test negative binomial model.
    # ps[0] - mean of nbinom distribution
    # ps[1] - aggregation k factor.
    '''
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

class ABC(object):
    
    def __init__(self):
        
        self.parameters = Parameters(tols,vs,xs)
        res = None
    ##
    # setup the ABC chain using wizard that guides you through process.
    # setup the ABC chain defining as many things as required.
    def setup(self,model_func=None,dist_func=None,tolerances=None,params=None,priors=None):
        '''TODO: Check data, check dist_func '''
        if tolerances: self.parameters.tolerances = tolerances
        if priors: self.parameters.vs = priors
        
    
    ##    
    def fit(self):
        '''
        # use dist_func to estimate distribution of errors from data xs
        # outputs tolerances.
        '''
        sample_size = 1000
        es = np.zeros(sample_size)
        for i in range(sample_size):
            ys = fakeSim(v1(),v2())
            es[i] = distFunc(ys,xs)
        es = es[np.isfinite(es)]
        if (es.size == 0):
            raise NameError('Couldn\'t find finite tolerance values. Priors must be way off.')
        tols = np.linspace(np.percentile(es,2.5),np.percentile(es,97.5),num=10)
        self.parameters.tols = tols[::-1]
        return self.parameters.tols
    ##
    def run(self,particle_num):
        '''
        # run the chain for n particles
        '''
        self.res = abcprcParralel(parameters=self.parameters,N=particle_num)
    ##
    def trace(self,plot=False):
        '''
        produce trace of partciles once chain has been run. inlude plot boolean
        '''
        if (self.res == None):
            raise NameError('No results. Use run() to generate posterior')
        else:
            # TODO: add plotting functionality. Ability to return parameter fits at different
            # tolerance levels.
            if plot == False:
                return self.res
            else:
                p1,p2 = self.res[0],self.res[1]
    
                x1 = pd.Series(p1[0,:], name="k")
                x2 = pd.Series(p2[0,:], name="m")
                plt.figure()
                g = sns.jointplot(x=x1, y=x2,kind="scatter")#,xlim=(0,0.2),ylim=(0,21))
                #g.ax_joint.plot(p0_true,p1_true,'ro')
                
    
                x1 = pd.Series(p1[5,:], name="k")
                x2 = pd.Series(p2[5,:], name="m")
                plt.figure()
                g = sns.jointplot(x=x1, y=x2,kind="scatter")#,xlim=(0,0.2),ylim=(0,21))
                #g.ax_joint.plot(p0_true,p1_true,'ro')
                
                
                x1 = pd.Series(p1[-1,:], name="k")
                x2 = pd.Series(p2[-1,:], name="m")
                plt.figure()
                g = sns.jointplot(x=x1, y=x2,kind="scatter")#,xlim=(0,0.2),ylim=(0,21))
                #g.ax_joint.plot(p0_true,p1_true,'ro')
            
    
    '''
    
    private functions
    
    '''

class Parameters(object):
    '''
    
    Defines all necessary parameters to run abc particles including:
        - particle number
        - priors
        - tolerances
        etc.
    
    '''    
    def __init__(self,tols,vs,xs):
        self.tols = tols
        self.particle_num = None #defined at run time so don't define now.
        self.vs = vs
        self.xs = xs #define data


##
# Main function to implement ABC-PRC scheme.
# input - tols : tolerances, N : no. of particles.
def abcprcParralel(parameters,N=N):
    #pool = Pool(processes=4) 
    #set-up matrices to record particles
    # structure is pRecs[i][j,k] i= parameter, j = time, k = particle.
    pRecs = []
    pRecs.append( np.zeros((parameters.tols.size,N)) )
    pRecs.append( np.zeros((parameters.tols.size,N)) )



    for t, tol in enumerate(parameters.tols):

        if (t==0):
            #initialise first particles from the priors v1,v2
            pRecs[0][0,:] = parameters.vs[0](size=N)
            pRecs[1][0,:] = parameters.vs[1](size=N)
        else:
            func = partial(particlesF,t,pRecs[0],pRecs[1],parameters.tols,parameters.xs)
            try:
                #vfunc = np.vectorize(func)                
                #res = pool.map(func, range(N))#vfunc(range(N))#
                res = Parallel(n_jobs=num_cores)(delayed(func)(i) for i in range(N))
            except KeyboardInterrupt:
                print 'got ^C while pool mapping'
       
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


def decorate(function):
    ''' 
    decorate a function to add an index, which is used to uniquely
    identify a run when running multiprocessing
    '''
    def wrap_function(*args, **kwargs):
        ii = None
        return function(ii,*args, **kwargs)
    return wrap_function
   
##
def distFunc(ys,xs):
    '''
    # Calculate distance between two empirical distributions.
    # input parameters for model.
    # output: distance between generated data and data xs.
    '''
    if (np.sum(ys)==0):
        return np.inf
    else:
        kernely = stats.gaussian_kde(ys)
        kernelx = stats.gaussian_kde(xs)
        xx = np.linspace(np.min(xs),np.max(xs)) #range over data.
        return stats.entropy(kernelx(xx),qk=kernely(xx)) #KL-divergence.


##
def particlesF(t,p1A,p2A,tols,xs,ii):
    '''
    # Filter particles step.
    '''
    
    np.random.seed()
    sys.stdout.flush()
    a_star = np.zeros(2)
    rho = tols[t]+1.
    N = tols.size
    n = 1000
    rejects = 0
    while(rho>tols[t] and rejects < n):
        #TODO: if no particles are accepted after a number of steps tolerance may be too low
        # fix by adding condition to raise error after n particles being rejected.
        r = np.random.randint(0,high=N)
        a_star[0] = stats.gamma.rvs(p1A[t-1,r]/rw_var,scale=rw_var)#p1A[t-1,r] + stats.norm.rvs(scale=0.1) 
        a_star[1] = stats.gamma.rvs(p2A[t-1,r]/rw_var,scale=rw_var)#p2A[t-1,r] + stats.norm.rvs(scale=0.1)
        ys = fakeSim(a_star[0],a_star[1],ii)        
        rho = distFunc(ys,xs)
        rejects += 1
        
    if (rejects >= n):
        raise NameError('Rejected all particles. Try increasing tolerances or increasing number of particles to reject.')
    p1 = a_star[0]
    p2 = a_star[1] 
    return p1,p2


    
    
if __name__ == '__main__':
    # initial test
    runSims = False
    if runSims:
        plt.hist(xs,bins=30)
        p = Parameters(tols,vs,xs)
        res = abcprcParralel(p)
    p1,p2 = res[0],res[1]
    
    x1 = pd.Series(p1[0,:], name="k")
    x2 = pd.Series(p2[0,:], name="m")
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