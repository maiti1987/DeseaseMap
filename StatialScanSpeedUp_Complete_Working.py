# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:14:37 2016

@author: aniruddha
"""

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

# logistic            
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
def calculaterisk(p):
    return -1 *np.log(1/p - 1)

#### Generate Dataset
NumRows=10
NumCols=10
backGroundProb=.001 
NumInstance=2000
DeseaseProbability=0.1
highRiskRows=[5,6,7]
highRiskColumns=[3,4,5]

# GenerateDataset
NumPlaces=NumRows*NumCols

# Assign  Back ground Desease Risks

backGroundrisk=calculaterisk(backGroundProb)
DeseaseRisks = [[backGroundrisk for x in range(NumCols)] for y in range(NumRows)] 


PlaceIds=[i for i in range(NumPlaces)]
splaceID=[i for i in range(NumPlaces)]

        
Individuals=[i for i in range(NumInstance)]

# Generate Location Data

LocationData=np.zeros((NumInstance, NumPlaces))

for inst in Individuals :
    
    NumvisitedUnits=random.randint(1, 2) # Number of Visited places
    #NumvisitedUnits=1
    
    random.shuffle(splaceID) 
    visitedPlaces=splaceID[0:NumvisitedUnits] # randomly select some places
    
    TotalTime=0
    x=[]
    for place in PlaceIds :
        if place in visitedPlaces :
            TimeSpent= random.randint(1, 500) # randomly assign some time
        else :
            TimeSpent=0
        x.append(TimeSpent)
        TotalTime=TotalTime+TimeSpent
    x_norm=[i/TotalTime for i in x] # Normalize
    LocationData[inst, :]=x_norm
    X=np.array(LocationData)
    

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
for x in LocationData:
    prob=sigmoid(x.dot(r_true))
    True_Prob.append(prob)
    s = np.random.uniform(0,1,1)
    if s < prob :
        Y.append(1)
    else :
        Y.append(0)
Y=np.array(Y)   

             
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
    m,n=X.shape
    indexList=get_indices(subregion, NumCols)
    xin1=X[np.ix_([person for person in range(m)],\
           indexList)]
    xin=sum(xin1.T)
    xin=xin.reshape(len(xin),1)
    Y1=Y.reshape(len(Y),1)
    digitized = np.digitize(xin, bins)
    
    cbp=[sum(Y1[digitized==i]) for i in range(1,len(bins)+1)]
    cbn=[len(Y1[digitized==i])-sum(Y1[digitized==i]) for i in range(1,len(bins)+1)]

    return cbp, cbn
    
    
# identify regions with no cases 
positivecases=[i for i,x in enumerate(Y) if x==1]
pLoc=sum(X[positivecases])
EmptyBlocksIndices=[i for i,x in enumerate(pLoc) if x==0]
LocMatrix=np.arange(NumRows*NumCols).reshape((NumRows, NumCols))
EmptyBlocks=[]
for blockind in EmptyBlocksIndices :
    KK=np.where(LocMatrix==blockind)
    p=KK[0][0]
    q=KK[1][0]
    EmptyBlocks.append((p,q))
    

# start clock
tic =time.time()

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



def isComputationRequired(subregion):
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
    if isComputationRequired(subregion) :
    
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
            print(Sr)
            SR_list.append(Sr)
            
        else :
            print(0)
            SR_list.append([0])
    else :
        print('Computation Not done')
        SR_list.append([0])
        
# Find out the maximum subregion
highRisksubregion=AllTestArea[SR_list.index(max(SR_list))]
#print('High Risk SubRegion', highRisksubregion)

# stop clock
toc=time.time()

print('Time elapsed', toc-tic)
print('High Risk SubRegion', highRisksubregion)
        

