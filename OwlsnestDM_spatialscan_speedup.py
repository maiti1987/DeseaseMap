
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 10:13:38 2016

@author: aniruddha
"""


# This file generates the simulated scenario
import random 
import numpy as np
import pickle
import time
import scipy.sparse as sp

#
import pdb
import sys
#import os.path


# logistic            
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
def calculaterisk(p):
    return -1 *np.log(1/p - 1)


# GenerateDataset


def GenerateDataset(NumRows, NumCols, backGroundProb, \
                    DeseaseProbability, NumInstance, highRiskRows, highRiskColumns):
                        
        NumPlaces=NumRows*NumCols
        # Assign  Back ground Desease Risks
        backGroundrisk=calculaterisk(backGroundProb)
        DeseaseRisks = [[backGroundrisk for x in range(NumCols)] for y in range(NumRows)] 
        
        
        PlaceIds=[i for i in range(NumPlaces)]
                
        Individuals=[i for i in range(NumInstance)]
        
        # Generate Location Data
        
        
        row=[]
        col=[]
        data=[]
        
        for inst in Individuals :
            
            NumvisitedUnits= random.randint(1, 8)# Number of Visited places
            #NumvisitedUnits=1
            
            pers=np.ones(NumvisitedUnits)*inst
            row=row + pers.tolist()
            visitedPlaces=random.sample(PlaceIds, NumvisitedUnits)
            col=col+ visitedPlaces
            timedata1=np.random.rand(NumvisitedUnits)
            timedata=[i/sum(timedata1) for i in timedata1]
            data=data + timedata
            
        X=sp.csc_matrix((data, (row, col)), shape=(NumInstance, NumPlaces))
            
            
            
            
        
        #X=np.array(LocationData)
        
        # Assign   Desease Probability
        
        deseaserisk=calculaterisk(DeseaseProbability)
        
        for row in highRiskRows :
            for col in highRiskColumns :
                   DeseaseRisks[row][col]=deseaserisk
        
        r_true=np.reshape(DeseaseRisks, NumPlaces) # true desease risks
        DeseaseRisks=np.array(DeseaseRisks)
        
        
        # create Target Vector according to the underlying desease risk
        Y=[]
        True_Prob=[]
        for x in X:
            prob=sigmoid(x.dot(r_true))
            True_Prob.append(prob)
            s = np.random.uniform(0,1,1)
            if s < prob :
                Y.append(1)
            else :
                Y.append(0)
        Y=np.array(Y) 
        
        return X,Y
             
 # end of data generation
     
# the following methods are required for predictions
    
def grad(X_temp, cbp, cbn, y_pred):
    
     grad=np.array([0,0])
     for b in range(len(y_pred)) :
         grad=grad + cbp[b] * (1-y_pred[b]) * X_temp[b, :] + cbn[b] * -1 * y_pred[b] * X_temp[b, :] 
     
    
     return  grad
     
def Hessian(X_temp, cbp, cbn, y_pred):
    hess=np.zeros((2, 2))
    for b in range(len(y_pred)) :
         hess=hess - (cbp[b]+cbn[b]) * y_pred[b] * (1-y_pred[b]) * np.dot(X_temp[b, :].reshape(2,1), X_temp[b, :].reshape(1,2))

    return hess

# prediction function
def predict(r, X, cutoff):
    m, n = X.shape
    pred = np.zeros(shape=(m, 1))

    h = sigmoid(np.dot(X,r))

    for it in range(0, h.shape[0]):
        if h[it] > cutoff:
            pred[it, 0] = 1
        else:
            pred[it, 0] = 0

    return pred
    
# given a list of tuples containing co-ordinates, 
#this method calculate place indices
            
def get_indices(cordinateList,NumCols):
    indexList=[]
    for cord in cordinateList :
        row=cord[0]
        col=cord[1]
        index=row*NumCols+col
        indexList.append(index)
    return indexList

def CalculateXin(subregion, X, NumCols):
    m,n=X.shape
    indexList=get_indices(subregion, NumCols)
    xin1=X[np.ix_([person for person in range(m)],\
           indexList)]
    xin=sum(xin1.T)
    xin=xin.reshape(len(xin),1)    
    return xin
    
def CalculateDiscretizeData(subregion, X, Y, NumCols, bins):
    try :
        m,n=X.shape
        indexList=get_indices(subregion, NumCols)
        if len(indexList)==1 :
            xin1=X.getcol(indexList[0])
        else :

            xin1=X[np.ix_([person for person in range(m)],\
               indexList)]
               
        xin=xin1.sum(axis=1)
        xin=xin.reshape(m,1)
        Y1=Y.reshape(len(Y),1)
        digitized = np.digitize(xin, bins)
        
        cbp=[sum(Y1[digitized==i]) for i in range(1,len(bins)+1)]
        cbn=[len(Y1[digitized==i])-sum(Y1[digitized==i]) for i in range(1,len(bins)+1)]
    except ValueError :
        print("Oops!  That was no valid number.  Try again...")
        #pdb.set_trace()

    return cbp, cbn
    
    
def isComputationRequired(subregion, EmptyBlocks):
        SS=np.array(subregion)
        
        size=int(np.sqrt(len(subregion)))
        
        if size==1 :
            if subregion[0] in EmptyBlocks:
                return False
            else:
                return True
        else :
            # col shift
            rightreg=[p for p in subregion if p[1]==max(SS[:,1])]
            if (min(SS[:,1])!=0) and (set(rightreg).issubset(set(EmptyBlocks))) :
                return False
            belowreg=[p for p in subregion if p[0]==max(SS[:,0])]
            if (min(SS[:,0])!=0) and (set(belowreg).issubset(set(EmptyBlocks))) :
                return False
            if (size % 2)==1 :
                outreg=[]
                outadd=[]
                outreg=[p for p in subregion if p[1]==max(SS[:,1])]
                outadd=[p for p in subregion if p[0]==max(SS[:,0])]
                outreg= outreg + outadd
                outadd=[]
                outadd=[p for p in subregion if p[0]==min(SS[:,0])]
                outreg= outreg + outadd
                outadd=[]
                outadd=[p for p in subregion if p[1]==min(SS[:,1])]
                outreg= outreg + outadd
                outadd=[]
                outreg=list(set(outreg))
                if (set(outreg).issubset(set(EmptyBlocks))) :
                    return False
        
        return True 
 

def CalculateHighRiskSubRegion(X,Y, NumRows, NumCols):   
    
        # identify regions with no cases 
        positivecases=[i for i,x in enumerate(Y) if x==1]
        posX=X[positivecases]
        nZval=np.diff(posX.indptr)
        EmptyBlocksIndices=[i for i,x in enumerate(nZval) if x==0]
        LocMatrix=np.arange(NumRows*NumCols).reshape((NumRows, NumCols))
        EmptyBlocks=[]
        for blockind in EmptyBlocksIndices :
            KK=np.where(LocMatrix==blockind)
            p=KK[0][0]
            q=KK[1][0]
            EmptyBlocks.append((p,q))
            
        
        
        
        # calculate the denominator for the objective of optimization function
        SR_list=[]
        C=float(sum(Y))
        N=float(len(Y))
        L0=  C*np.log(C) + (N-C)*np.log(N-C)  -  N * np.log(N)
        Denom=L0
        
        # Calculate all possible square regions
        
        AllTestArea=[]
        for i in range(min(NumRows,NumCols)-1) :
        #for i in range(2,3) :
            sqsize=i+1
            for srows in range(NumRows-sqsize+1):
                for scols in range(NumCols-sqsize+1):
                    TestArea=[(rows,cols) for rows in range(srows,srows+sqsize) for cols in range(scols,scols+sqsize)]
                    AllTestArea.append(TestArea)





        # set up bins
        M=100
        bins = np.linspace(0, 1, M+1)
        middle=(bins[1]-bins[0]) / 2
        xin=[bins[i]+middle for i in range(len(bins))] 
        xin=np.array(xin)
        xin=xin.reshape(len(xin),1)
        xout=1-xin
        X_temp=np.concatenate((xin, xout), axis=1)


        # for each of the regions compute log likelihood ratio
        for subregion in AllTestArea :
            if isComputationRequired(subregion, EmptyBlocks) :
            
                cbp,cbn=CalculateDiscretizeData(subregion,X,Y,NumCols,bins)
                Ytar=[cbp[i]/float(cbp[i]+cbn[i]) if not (cbp[i]+cbn[i]==0) else 0 for i in range(len(cbp))]
                r=np.random.rand(2,1)
                #eta=1  # Not required
                olderr=float("inf")
                errReduct=float("inf")    
                flag=0
                for i in range(10) :
                    if flag==0 :
                    # calculate gradient and Hessian
                        y_p= sigmoid(np.dot(X_temp,r))
                        y_pred=np.array([i[0] for i in y_p])
                        err1= y_pred-Ytar
                        err=np.sqrt(sum(n*n for n in err1))/len(err1)
                        g=grad(X_temp, cbp, cbn, y_pred)
                        if olderr==err :
                            flag =1
                        #print('iteration :  ', i, '  Error  : ', err)
                        H=Hessian(X_temp, cbp, cbn, y_pred)
                        STEP=np.dot(np.linalg.inv(H), g)
                        r=r-STEP.reshape(2,1)
                        errReduct=olderr-err
                        olderr=err
                
                
                if r[0] > r[1] :
                    y_p = sigmoid(np.dot(X_temp,r))
                    maxNumerator=0
                    for b in range(len(y_p)):
                        maxNumerator=maxNumerator + cbp[b]*np.log(y_p[b]) + cbn[b] * np.log(1-y_p[b])
                       
                    Sr=maxNumerator-Denom
                    #print(Sr)
                    SR_list.append(Sr)
                    
                else :
                    #print(0)
                    SR_list.append([0])
            else :
                #print('Computation Not done')
                SR_list.append([0])
        
        # Find out the maximum subregion
        highRisksubregion=AllTestArea[SR_list.index(max(SR_list))]
        #print('High Risk SubRegion', highRisksubregion)
        return highRisksubregion

#### Generate Dataset params
NumRows=int(sys.argv[1]) 
NumCols=int(sys.argv[2])
NumInstance=int(sys.argv[3])
backGroundProb=float(sys.argv[4])
DeseaseProbability=float(sys.argv[5])





highRiskRows=[3,4,5]
highRiskColumns=[5,6,7]
NumPlaces=NumRows*NumCols

#pdb.set_trace()
# start clock
tic =time.time()
#print('Generating Dataset...')
X,Y=GenerateDataset(NumRows, NumCols, backGroundProb, \
    DeseaseProbability, NumInstance, highRiskRows, highRiskColumns)

# stop clock
toc=time.time()

datagentime=toc-tic

    
# start clock
tic =time.time()    
highRisksubregion = CalculateHighRiskSubRegion(X,Y, NumRows, NumCols)  
# stop clock
toc=time.time()
Computationtime=toc-tic

#print('Time elapsed', toc-tic)
#print('High Risk SubRegion', highRisksubregion)

#open the file to write the result
f=open('Result.txt', 'a')
f.write("\n\nParameter List.....\n")
f.write("NumRows : {0} \n".format(NumRows))
f.write("NumCols : {0} \n".format(NumCols))
f.write("NumInstance : {0} \n".format(NumInstance) )
f.write("backGroundProb : {0}\n".format(backGroundProb) )
f.write("DeseaseProbability : {0}\n".format(DeseaseProbability) )
f.write("....................................\n")
f.write("data generation time {0}\n".format(datagentime) )
f.write("Result.....\n\n\n")
f.write('Computation time : {0}\n'.format(Computationtime) )
f.write("Identified Subregion \n")
for i,x in enumerate(highRisksubregion):
   f.write('{0:3d}. {1:3d} {2:3d}\n'.format(1, x[0], x[1]) )
   
f.write("\n ################################################### \n")
f.close()
        

