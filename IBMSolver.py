#individual-based model solver
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as sts
import subprocess
import random as random
import string as string
import os as os
from multiprocessing import Pool
from joblib import Parallel, delayed  
import multiprocessing
import time
sb.set_context("paper",font_scale=1.4)
model_path = './'

def paramInput(data,valueR,index): #changes parameter by ratio defined in valueR
    
    newparam = data[index].split(' : ')
    newparam[0] = str(valueR*float(newparam[0])) # change V:H to whatever.
    data[index] = ' : '.join(newparam)
    return data

def paramInputFix(data,param,index): #changes parameter by ratio defined in valueR
    
    newparam = data[index].split(' : ')
    newparam[0] = str(param) # change V:H to whatever.
    data[index] = ' : '.join(newparam)
    return data
    
'''set parameters '''
tsiz=121
yrs = 100.0
yr2equib = 10
n = 200
dt = 1
nparams = 6 #number of system parameters
oneyr = 6 #snapshot taken every two months.Hence every six time-points is a year.
reps = 20
index = 19 #index of parameter going to change 17= sig-death rate mosquito, 16 = g, 19 - psi2
lowR = 0.7 #decrease of parameter
highR = 0.9 #increase of parameter
ages = np.linspace(0,80,11)
#ages = np.array([5,8,11,14,19,24,29,34,44,54,70])
num_cores = multiprocessing.cpu_count()

def mfOutSim(k,v_to_h,i):
    with open(model_path+'parameters.txt', 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    newdata=data[:] # colon operator to make a copy and not edit original.
    newdata = paramInputFix(newdata,k,3)
    newdata = paramInputFix(newdata,v_to_h,9)
    param_text = 'parameters' + ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
        # and write everything back
    with open(model_path+param_text, 'w') as file:
        file.writelines( newdata )
        
    if k<0.05:
        mf = np.zeros((200,n))
    else:
        mfp, wp, mf, w = runSimMfICTPrevInts(param_text,i)
    os.remove(param_text)
    return mf[-120,:]

def runMultipleSims(mu1=1.0,mu2=1.0,k=0.1,v_to_h=7,
                    nMDA=0,coverage=0.65,mdaFreq=12,lbdaR=1.0,v_to_hR=1.0,vecComp=1.0,
                    vecCap=1.0,vecD=1.0,ageProfile=True,mfAgeProfile=False,ICTPrevs=False,
                    mfICTInts = False,
                    numReps=20,species=0,sysComp=0.0,rhoBComp=0.0,rhoCN=0.0,aWol=0,covN=0.0,
                    chi=1.0,tau=0.55,
                    IDAInt=0,IDAchi =1.0, IDAtau=1.0, IDAIntroYrs=5.0):
    with open(model_path+'parameters.txt', 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    s2 = 0.00275

    newdata=data[:] # colon operator to make a copy and not edit original.
    newdata = paramInputFix(newdata,mu1,0)
    newdata = paramInputFix(newdata,mu2,1)
    newdata = paramInputFix(newdata,mu2,2)  
    newdata = paramInputFix(newdata,k,3)
    newdata = paramInputFix(newdata,v_to_h,9)
    newdata = paramInputFix(newdata,lbdaR,21)
    newdata = paramInputFix(newdata, v_to_hR ,22)
    newdata = paramInputFix(newdata,nMDA,23)
    newdata = paramInputFix(newdata,mdaFreq,24)
    newdata = paramInputFix(newdata,coverage,25)
    newdata = paramInputFix(newdata,s2,26)
    newdata = paramInput(newdata,vecComp,19)
    newdata = paramInput(newdata,vecCap,16)
    newdata = paramInput(newdata,vecD,17)
    
    newdata = paramInputFix(newdata,chi,27) #proportion of mf removed for MDA
    newdata = paramInputFix(newdata,tau,28) #proportion of worms sterilised
    newdata = paramInputFix(newdata,int(species),30)    # 0 - Anopheles, 1- Culex
    newdata = paramInputFix(newdata,sysComp,29)
    newdata = paramInputFix(newdata,rhoBComp,31)
    newdata = paramInputFix(newdata,int(aWol),32)

    newdata = paramInputFix(newdata,covN,34)
    newdata = paramInputFix(newdata,rhoCN,36)
    newdata = paramInputFix(newdata,int(IDAInt),37)
    newdata = paramInputFix(newdata,IDAchi,38)
    newdata = paramInputFix(newdata,IDAtau,39)
    newdata = paramInputFix(newdata,IDAIntroYrs,40)
    #newdata = paramInput(newdata,lowR,index) # sig mosquito death rate 17= sig-death rate mosquito, 16 = g, 19 - psi2

    # and write everything back
    with open(model_path+'parameters.txt', 'w') as file:
        file.writelines( newdata )
    #ms = np.zeros((np.size(ages),reps))
    indexes = range(numReps)    
    #pool = Pool()
    #results = pool.map(runSimAgePrev, indexes)
    if ageProfile and not mfAgeProfile and not ICTPrevs and not mfICTInts:
        results = Parallel(n_jobs=num_cores)(delayed(runSimAgePrev)(i) for i in indexes) 
    elif not ageProfile and not mfAgeProfile and not ICTPrevs and not mfICTInts:
        results = Parallel(n_jobs=num_cores)(delayed(runSimNoAgePrev)(i) for i in indexes)
    elif ICTPrevs:
        results = Parallel(n_jobs=num_cores)(delayed(runSimNoAgePrevICT)(i) for i in indexes)
    elif mfICTInts:
        results = Parallel(n_jobs=num_cores)(delayed(runSimMfICTPrevInts)(i) for i in indexes)
    else:
        results = Parallel(n_jobs=num_cores)(delayed(runSimMfCount)(i) for i in indexes)
    '''re-write parameters back to file'''
    with open(model_path+'parameters.txt', 'w') as file:
        file.writelines( data ) 
        
    return results
    #for i in range(reps): 
    #    print str(i)
    #    filename = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
    #    filename = filename+'.txt'
    #    ts, mfp, wp, ind_ages = runSim(mu1=mu1,mu2=mu2,k=k,v_to_h=v_to_h,nMDA=nMDA,coverage=coverage,mdaFreq=mdaFreq,lbdaR=lbdaR,v_to_hR=v_to_hR,vecComp=vecComp,vecCap=vecCap,vecD=vecD,filename=filename)
    #    tages,mmff,mtot = agePrevalence(mfp,ind_ages,bins=ages)  
    #    ms[:,i] = mmff
    #return mmff
def runSimAgePrev(index):
    filename = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
    filename = '0'+filename+'.txt'
    ts, mfp, wp, ind_ages = runSimNoInput(filename=filename,index=index)
    tages,mmff,mtot = agePrevalence(mfp,ind_ages,bins=ages)  
    return mmff
def runSimMfCount(index):
    filename = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
    filename = '0'+filename+'.txt'
    ts, mfp, wp, ind_ages, mf = runSimNoInput(filename=filename,finalDistribution=True,mfDistribution=True,index=index) 
    return mf, ind_ages
def runSimNoAgePrev(index):
    filename = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
    filename = '0'+filename+'.txt'
    ts, mfp, wp, ind_ages = runSimNoInput(filename=filename,finalDistribution=False,index=index) 
    return mfp
def runSimNoAgePrevICT(index):
    filename = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
    filename = '0'+filename+'.txt'
    ts, mfp, wp, ind_ages = runSimNoInput(filename=filename,finalDistribution=False,index=index) 
    return mfp,wp
def runSimMfICTPrevInts(param_text,index): #get prevalences and intensity of both ICT and mf.
    filename = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
    filename = '0'+filename+'.txt'
    ts, mfp, wp, ind_ages,mf,w = runSimNoInput(filename=filename,finalDistribution=False,allDistribution=True,index=index,param_text=param_text) 
    return mfp, wp, mf, w
def runSim(mu1=1.0,mu2=1.0,k=0.1,v_to_h=7,
                    nMDA=0,coverage=0.65,mdaFreq=12,lbdaR=1.0,v_to_hR=1.0,vecComp=1.0,
                    vecCap=1.0,vecD=1.0,
                    species=0,sysComp=0.0,rhoBComp=0.0,rhoCN=0.0,aWol=0,covN=0.0,
                    chi=1.0,tau=0.55,filename='test3.txt'):
    '''change model parameters using txt file '''
    with open(model_path+'parameters.txt', 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    s2 = 0.00275

    newdata=data[:] # colon operator to make a copy and not edit original.
    newdata = paramInputFix(newdata,mu1,0)
    newdata = paramInputFix(newdata,mu2,1)
    newdata = paramInputFix(newdata,mu2,2)  
    newdata = paramInputFix(newdata,k,3)
    newdata = paramInputFix(newdata,v_to_h,9)
    newdata = paramInputFix(newdata,lbdaR,21)
    newdata = paramInputFix(newdata, v_to_hR ,22)
    newdata = paramInputFix(newdata,nMDA,23)
    newdata = paramInputFix(newdata,mdaFreq,24)
    newdata = paramInputFix(newdata,coverage,25)
    newdata = paramInputFix(newdata,s2,26)
    newdata = paramInput(newdata,vecComp,19)
    newdata = paramInput(newdata,vecCap,16)
    newdata = paramInput(newdata,vecD,17)
    
    newdata = paramInputFix(newdata,chi,27) #proportion of mf removed for MDA
    newdata = paramInputFix(newdata,tau,28) #proportion of worms sterilised
    newdata = paramInputFix(newdata,int(species),30)    # 0 - Anopheles, 1- Culex
    newdata = paramInputFix(newdata,sysComp,29)
    newdata = paramInputFix(newdata,rhoBComp,31)
    newdata = paramInputFix(newdata,int(aWol),32)

    newdata = paramInputFix(newdata,covN,34)
    newdata = paramInputFix(newdata,rhoCN,36)
    # and write everything back
    with open(model_path+'parameters.txt', 'w') as file:
        file.writelines( newdata )
     

    '''run model '''
    args = str(yrs) + " " + str(n) + " " + str(filename)
    cmd = model_path+'timer'
    ccmd = cmd + " " + args
    subprocess.call(ccmd,shell=True)
    '''import data '''
    outtm = np.loadtxt(model_path + filename)

    wp = (outtm[0::6,:]+outtm[0::6,:])
    wp = wp[-1,:]
    mfp = outtm[2::6,:]
    mfp = mfp[-1,:]
    ages = outtm[-2,:]/12.0
    #after running model return everything back to normal with original data.
    # and write everything back
    with open(model_path+'parameters.txt', 'w') as file:
        file.writelines( data ) 
    os.remove(filename)
    ts = np.concatenate((np.linspace(0,100,100),np.linspace(100,120,120)))
    return ts, mfp, wp, ages
    
def runSimIntervention(mu1=1.0,mu2=0.29,k=0.32,v_to_h=7,nMDA=0,coverage=0.65,mdaFreq=12,lbdaR=1.0,v_to_hR=1.0,vecComp=1.0,vecCap=1.0,vecD=1.0,filename='test3.txt',t=120):
    '''change model parameters using txt file '''
    with open(model_path+'parameters.txt', 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    newdata=data[:] # colon operator to make a copy and not edit original.
    newdata = paramInputFix(newdata,mu1,0)
    newdata = paramInputFix(newdata,mu2,1)
    newdata = paramInputFix(newdata,mu2,2)  
    newdata = paramInputFix(newdata,k,3)
    newdata = paramInputFix(newdata,v_to_h,9)
    newdata = paramInputFix(newdata,lbdaR,21)
    newdata = paramInputFix(newdata, v_to_hR ,22)
    newdata = paramInputFix(newdata,nMDA,23)
    newdata = paramInputFix(newdata,mdaFreq,24)
    newdata = paramInputFix(newdata,coverage,25)

    newdata = paramInput(newdata,vecComp,19)
    newdata = paramInput(newdata,vecCap,16)
    newdata = paramInput(newdata,vecD,17)
    #newdata = paramInput(newdata,lowR,index) # sig mosquito death rate 17= sig-death rate mosquito, 16 = g, 19 - psi2

    # and write everything back
    with open(model_path+'parameters.txt', 'w') as file:
        file.writelines( newdata )
     

    '''run model '''
    args = str(t) + " " + str(n) + " " + str(filename)
    cmd = model_path+'timer'
    ccmd = cmd + " " + args
    subprocess.call(ccmd,shell=True)
    '''import data '''
    outtm = np.loadtxt(model_path + filename)

    wp = (outtm[0::6,:]+outtm[0::6,:])>0
    
    mfp = outtm[2::6,:]>1.0
    
    ages = outtm[-2,:]/12.0
    #after running model return everything back to normal with original data.
    # and write everything back
    with open(model_path+'parameters.txt', 'w') as file:
        file.writelines( data ) 
    os.remove(filename)
    ts = np.concatenate((np.linspace(0,100,100),np.linspace(100,120,120)))
    return ts, mfp, wp, ages

def runSimNoInput(filename='test3.txt',finalDistribution=True,mfDistribution=False,allDistribution=False,index=0,param_text = 'parameters.txt'):
    ''' use for parrellisation as does not change the parameter file. '''
    '''run model '''
    if finalDistribution:
        args = str(yrs) + " " + str(n) + " " + str(filename) + " " + str(index) + " " + str(param_text)
    else:
        args = str(120.0) + " " + str(n) + " " + str(filename) + " " + str(index) + " " + str(param_text)
    cmd = model_path+'timer'
    ccmd = cmd + " " + args
    
    subprocess.call(ccmd,shell=True)
    '''import data '''
    
    outtm = np.loadtxt(model_path + filename)

    wp = (outtm[0::6,:]+outtm[0::6,:])>0
    w = outtm[0::6,:]+outtm[0::6,:]
    mfp = outtm[2::6,:]>1.0
    mf = outtm[2::6,:]
    ages = outtm[-2,:]/12.0
    if finalDistribution:
        wp = wp[-1,:]
        mfp = mfp[-1,:]
        mf = mf[-1,:]
    os.remove(filename)
    
        #mfp = np.zeros((200,200,200))
        #wp = np.zeros((200,200,200))
        #ages = np.zeros((200,200,200))
        #mf = np.zeros((200,200,200))
        #w = np.zeros((200,200,200))
    ts = np.concatenate((np.linspace(0,100,100),np.linspace(100,120,120)))
    if mfDistribution:
        return ts, mfp, wp, ages, mf
    elif allDistribution:
        return ts, mfp, wp, ages, mf, w
    else:
        return ts, mfp, wp, ages


def importData():
    '''import data '''
    outtm = np.loadtxt(model_path + 'test3.txt')

    mff = outtm[(2+nparams*oneyr*(100)-6)::nparams,:]
    mff = (mff>1.0)#mff*(mff>0.0) #>50.0
    mff = np.mean(mff,axis=1)

    wf = outtm[(0+nparams*oneyr*(100)-6)::nparams,:] + outtm[(1+nparams*oneyr*(100)-6)::nparams,:]
    wf = (wf>0.0)
    wf = np.mean(wf,axis=1)

    lf = outtm[(3+nparams*oneyr*(100)-6)::nparams,:]
    #lf = (lf>0.0)
    lf = np.mean(lf,axis=1)
    ts=np.linspace(0,20,121)
    return ts, mff, wf, lf
    
def importAgePrevalenceData():
    '''import data '''
    outtm = np.loadtxt(model_path + 'test3.txt')

    #mff = outtm[(2+nparams*oneyr*(yr2equib)-6),:]
    #mff = (mff>1.0)#mff*(mff>0.0) #>50.0
    #mff = np.mean(mff,axis=1)

    #wf = outtm[(0+nparams*oneyr*(yr2equib)-6),:] + outtm[(1+nparams*oneyr*(yr2equib)-6),:]
    #wf = (wf>0.0)
    
    wp = (outtm[0::6,:]+outtm[0::6,:])>0
    mfp = outtm[2::6,:]>1.0
    ages = outtm[4::6,:]
    agesf = outtm[4+nparams*oneyr*(yr2equib)-6,:]
    #wf = np.mean(wf,axis=1)

    #lf = outtm[(3+nparams*oneyr*(yr2equib)-6)::nparams,:]
    
    binsmf, mmff = agePrevalence(mfp[-1],ages[-1],binn=10)
    binsICT, ICTf = agePrevalence(wp[-1],ages[-1],binn=10)
    #lf = (lf>0.0)
    
    
    return binsmf/12.0, mmff,binsICT/12.0,ICTf

def agePrevalence(mff,agesf,binn=10,bins=np.array([]),prop=True):
    if np.size(bins)>0:
        binn = np.size(bins)
    mmff = np.zeros(binn)
    p_tot = np.zeros(binn)
    mmfferru = np.zeros(binn)
    mmfferrl = np.zeros(binn)
    if np.size(bins)==0:
        hist, bins = np.histogram(agesf,bins=binn)
    inds = np.digitize(agesf, bins)
    for i in np.arange(1,binn):
        n = np.sum(inds==i)
        if prop:
            p =  np.mean(mff[inds==i]>0)  #* 0.18#definitely right
            p_tot[i] = np.sum(inds==i)
            mmff[i] = p#np.mean(mff[inds==i]>0) * 0.5
            mmfferrl[i],mmfferru[i] = sts.binom.interval(0.95, n, mmff[i]) / n#binom_interval(p, n)
        else:
            p =  np.sum(mff[inds==i]>0)  #* 0.18#definitely right
            p_tot[i] = np.sum(inds==i)
            mmff[i] = p#np.mean(mff[inds==i]>0) * 0.5
    ages = bins[0:-2]+(bins[1:-1]-bins[0:-2])*0.5 
    return ages, mmff, p_tot
    
if __name__ == '__main__':
    t0 = time.time()
    results = runMultipleSims(mu1=1.0,mu2=1.0,k=0.065,v_to_h=0.8,nMDA=0,coverage=0.65,mdaFreq=12,lbdaR=1.0,v_to_hR=1.0,vecComp=1.0,vecCap=1.0,vecD=1.0)
    t1 = time.time()
    print str(t1-t0) + 's'
    
    for r in results:
        plt.plot(ages,r,color='0.8')
    plt.plot(ages,np.mean(results,axis=0),label='multiple simultions')
    n = 4000
    t0 = time.time()
    ts, mfp, wp, ind_ages = runSim(mu1=1.0,mu2=1.0,k=0.065,v_to_h=0.8,nMDA=0,coverage=0.65,mdaFreq=12,lbdaR=1.0,v_to_hR=1.0,vecComp=1.0,vecCap=1.0,vecD=1.0,filename='test3.txt')
    tages,mmff,mtot = agePrevalence(mfp,ind_ages,bins=ages)      
    t1 = time.time()
    print str(t1-t0) + 's'  
    plt.plot(ages,mmff,'r--',label='large simulation')
    plt.legend()
    plt.xlabel('age')
    plt.ylabel('probability')
    #plt.savefig('EnsembleLargeComparison.pdf')
def nRoundsCalc(mfps):
        prevs = np.mean(mfps[:,-120:,:],axis=2) #(reps,ts)
        tindexes = [np.where(p<=0.01) for p in prevs]
        tminindexes = np.array([])
        ts = np.linspace(0,20,120)
        for row in tindexes:
            if row[0].any():
                tminindexes = np.append(tminindexes,row[0][0])
            else:
                tminindexes = np.append(tminindexes,119)
        tminindexes = tminindexes.astype(int)
        trounds = ts[tminindexes]
        nrounds = np.floor(trounds)-1 
        return nrounds