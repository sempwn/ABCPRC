"""
Implementation of ABC-PRC code
reference: Parameter inference in small world network disease models with approximate Bayesian Computational methods
Walker et al. Physica A 2010
Created on Thu March 7 14:33:33 2016

@author: Mike Irvine (Sempwn)
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







#default parameters
num_cores = multiprocessing.cpu_count()
tols = np.array([1000,500,250,190,150,125,100,75,50,45,40,35,30])/1000.
N = 10
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
sim = nbinSim#ibm.mfOutSim#
#input data.
xs = sim(p0_true,p1_true,0)


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

        self.parameters = Parameters(tols,vs,xs,sim,distFunc)
        self.res = None
        self.acc_dists = None
    ##
    def setup(self,modelFunc=None,distFunc=None,tolerances=None,xs=None,priors=None):
        '''
        setup the ABC chain using wizard that guides you through process.
        setup the ABC chain defining as many things as required.
        '''
        '''TODO: Check data, check dist_func '''
        if (tolerances!=None): self.parameters.tolerances = tolerances
        if (priors!=None): self.parameters.vs = priors
        if (modelFunc!=None): self.parameters.sim = modelFunc
        if (distFunc!=None): self.parameters.distFunc = distFunc
        if (xs!=None): self.parameters.xs = xs


    ##
    def fit(self,sample_size = 1000):
        '''
        use dist_func to estimate distribution of errors from data xs
        outputs tolerances.
        '''

        es = np.zeros(sample_size)
        for i in range(sample_size):
            ps = [v() for v in self.parameters.vs]
            ys = self.parameters.sim(*ps)
            es[i] = self.parameters.distFunc(ys,self.parameters.xs)
            update_progress(float(i+1)/sample_size)
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
        self.res,self.acc_dists = abcprcParralel(parameters=self.parameters,N=particle_num)

    def paramMAP(self):
        '''
        return the maximum a posteriori for each parameter once run.
        Currently just using the mean. This needs to be improved.
        '''
        if self.res == None:
            raise NameError('Need to run first before returning results')
        results = []
        for r_param in self.res:
            results.append(np.mean(r_param[-1,:]))
        return results

    def fitSummary(self,percentiles=[2.5,97.5]):
        '''
        return the maximum a posteriori for each parameter once run.
        Currently just using the mean. This needs to be improved.
        '''
        if self.res == None:
            raise NameError('Need to run first before returning results')
        results = {'p':[],'lc':[],'uc':[]}
        for i,r_param in enumerate(self.res):
            p = np.median(r_param[-1,:])
            lc = np.percentile(r_param[-1,:],percentiles[0])
            uc = np.percentile(r_param[-1,:],percentiles[1])
            results['p'].append(p)
            results['lc'].append(lc)
            results['uc'].append(uc)
            print 'param {} : {} ({},{}) '.format(i,p,lc,uc)
        return results

    ##
    def trace(self,plot=False,tol=-1):
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
                matPlot(self.res,tol=tol)

    def save(self,filename):
        '''
        save main results from runs to file as .npz. Use load to load file.
        '''
        if (self.res == None):
            raise NameError('Can\'t save without results. Use run() first before saving.')

        np.savez(filename,res=self.res,acc_dists=self.acc_dists,
                 tolerances=self.parameters.tols)

    def load(self,filename):
        '''
        load file that's been formatted using save.
        '''
        if os.path.isfile(filename+'.npz'):
            out = np.load(filename+'.npz')
        else:
            raise NameError('Filename does not exist.')

        self.res = out['res']
        self.acc_dists = out['acc_dists']
        self.parameters.tols = out['tolerances']


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
    def __init__(self,tols,vs,xs,sim,distFunc):
        self.tols = tols
        self.particle_num = None #defined at run time so don't define now.
        self.vs = vs
        self.xs = xs #define data
        self.sim = sim #define simulation function.
        self.distFunc = distFunc


##
# Main function to implement ABC-PRC scheme.
# input - tols : tolerances, N : no. of particles.
def abcprcParralel(parameters,N=N):
    #set-up matrices to record particles
    # structure is pRecs[i][j,k] i= parameter, j = time, k = particle.
    pRecs = []
    for i in range(len(parameters.vs)):
        pRecs.append( np.zeros((parameters.tols.size,N)) )
    dist_acc = np.zeros((parameters.tols.size,N)) #records distances accepted for each particle.
    p_num = len(pRecs)

    for t, tol in enumerate(parameters.tols):

        if (t==0):
            #initialise first particles from the priors v1,v2
            for i in range(p_num):
                pRecs[i][0,:] = parameters.vs[i](size=N)

        else:
            parFunc = partial(particlesF,t,pRecs,parameters.tols,parameters.xs,
                              parameters.sim,parameters.distFunc)
            try:
                res = Parallel(n_jobs=num_cores)(delayed(parFunc)(i) for i in range(N))
            except KeyboardInterrupt:
                print 'got ^C while pool mapping'

            except Exception, e:
                print 'got exception: %r' % (e,)

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                traceback.print_exc()
            for i in range(p_num):
                pRecs[i][t,:] = np.array([a[i] for a in res]).flatten()
            dist_acc[t,:] = np.array([a[p_num] for a in res]).flatten()
        update_progress(float(t)/len(tols))


    res = pRecs
    update_progress(10.)
    sys.stdout.write("\n")
    return res, dist_acc

def ABCPRC(tols=tols,N=N): #deprecated. Should probably remove, unless we want a non-paralleled option?
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
def particlesF(t,pRecs,tols,xs,sim,distFunc,ii):
    '''
    # Filter particles step.
    '''
    p_num = len(pRecs)
    np.random.seed()
    sys.stdout.flush()
    a_star = np.zeros(p_num)
    rho = tols[t]+1.
    N = np.size(pRecs[0],axis=1) #number of particles. Should check all pRec match.
    n = 1000 #upper bound for number of samles rejected before raising error.
    rejects = 0 #count number of rejects.
    while(rho>tols[t] and rejects < n):
        #FIXED: if no particles are accepted after a number of steps tolerance may be too low
        # fix by adding condition to raise error after n particles being rejected.
        r = np.random.randint(0,high=N)
        for i in range(p_num):
            a_star[i] = stats.gamma.rvs(pRecs[i][t-1,r]/rw_var,scale=rw_var)#p1A[t-1,r] + stats.norm.rvs(scale=0.1)
        ys = sim(*a_star)#sim(a_star[0],a_star[1],ii)
        rho = distFunc(ys,xs)
        rejects += 1

    if (rejects >= n):
        raise NameError('Rejected all particles. Try increasing tolerances or increasing number of particles to reject.')

    res = a_star.tolist()
    res.append(rho)
    return res #return parameters and accepted distance.

def matPlot(res,tol=-1):
    pLen = len(res)
    plt.close('all')
    f, ax = plt.subplots(pLen,pLen)
    ind = 0
    ranges=[]
    for i in range(pLen):
        ranges.append([np.min(res[i][tol,:]),np.max(res[i][tol,:])])

    for i in range(pLen):
        for j in range(pLen):
            if (i == j):
                ax[i][j].hist(res[i][tol,:],range=ranges[i])
            else:
                ax[i][j].plot(res[j][tol,:],res[i][tol,:],'bo')
                ax[i][j].set_xlim(ranges[j])
                ax[i][j].set_ylim(ranges[i])
            if i!=(pLen-1):
                plt.setp(ax[i][j].get_xticklabels(),visible=False)
            if j!=0:
                plt.setp(ax[i][j].get_yticklabels(),visible=False)
            ind += 1
