
# coding: utf-8

# In[50]:

import numpy as np
from numpy import genfromtxt
from numpy import linalg as lg
from numpy import linalg as LA
from matplotlib import pyplot as plt
import pandas as pd
get_ipython().magic(u'matplotlib inline')

#Q1 Plot V2 vs V1.  Do you see a clear separation of the raw data?
data_1 = genfromtxt("C:/Users/vyoms/Desktop/dataset_1.csv", delimiter=",")
data = data_1[1:,:]
data_V1= data_1[1:,0]
data_V2= data_1[1:,1]
data_label = data_1[1:,2]

plt.scatter(data_V1[0:30], data_V2[0:30], marker = 'o', color = 'purple', alpha = 0.5)
plt.scatter(data_V1[30:60],data_V2[30:60], marker = 'o', color = 'purple', alpha = 0.5)
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('V2 vs V1')
plt.show()




def d_PCA(x):
    columnMean = x.mean(axis=0)
    columnMeanAll = np.tile(columnMean, reps=(x.shape[0], 1))
    xMeanCentered = x - columnMeanAll
#use mean_centered data or standardized mean_centered data
    
    dataForPca = xMeanCentered

#get covariance matrix of the data
    covarianceMatrix = np.cov(dataForPca, rowvar=False)
    print(covarianceMatrix)
#eigendecomposition of the covariance matrix
    eigenValues, eigenVectors = LA.eig(covarianceMatrix)
    II = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[II]
    eigenVectors = eigenVectors[:, II]
#get scores
    pcaScores = np.matmul(dataForPca, eigenVectors)
#collect PCA results
    pcaResults = {'data': x,
                         'mean_centered_data': xMeanCentered,
                         'PC_variance': eigenValues,'loadings': eigenVectors,
                         'scores': pcaScores}
    
    
    return pcaResults

myPCAResults = d_PCA(data)

V1 = data_V1
V2 = data_V2

# Total no. of X values
m = len(V1)

# Mean of X and Y values
mean_V1 = np.mean(V1)
mean_V2 = np.mean(V2)

# calculating b1 and b2
numerator = 0
denominator = 0
for i in range(m):
    numerator += (V1[i] - mean_V1) * (V2[i] - mean_V2)
    denominator += (V1[i] - mean_V1) ** 2
b1 = numerator / denominator
b0 = mean_V2 - (b1 * mean_V1)

max_V1 = np.max(V1) 
min_V1 = np.min(V1)

# Calculating line values x and y
v1 = np.linspace(min_V1, max_V1, 1000)
v2 = b0 + b1 * v1

plt.title('Regression Plot')
plt.scatter(data_V1[0:30], data_V2[0:30], marker = 'o', color = 'purple', alpha = 0.5)
plt.scatter(data_V1[30:60],data_V2[30:60], marker = 'o', color = 'purple', alpha = 0.5)
plt.plot([0,100*myPCAResults['loadings'][0,0]], [0, 100*myPCAResults['loadings'][1,0]],
            color='orange', linewidth=3)
plt.xlim(0, 40), plt.ylim(0, 40)


plt.show()
#b0
#b1

df = pd.read_csv('C:/Users/vyoms/Desktop/dataset_1.csv',header=None)

df1 = df.drop(df.index[0])

df2 = df1.drop(df.columns[2], axis=1)

df2 = df2.astype(float)

df3 = df2

df3_1 = df2.values[0:30,:]
df3_2 = df2.values[30:, : ]

df3_1 = df3_1.astype(float)
df3_2 = df3_2.astype(float)

m_1 = df3_1.mean(axis = 0)
m_2 = df3_2.mean(axis = 0)
mean_all = df2.mean(axis = 0)

mean_1 = m_1.reshape(1,2)
mean_1 = np.repeat(mean_1,30,axis = 0)

mean_2 = m_2.reshape(1,2)
mean_2 = np.repeat(mean_2,30,axis = 0)

within_class_scatter = np.zeros((2,2))
wcs_1 = np.zeros((2,2))
wcs_1 = np.matmul((np.transpose(df3_1 - mean_1 )), (df3_1 - mean_1))

wcs_2 = np.zeros((2,2))
wcs_2 = np.matmul((np.transpose(df3_2 - mean_2 )), (df3_2 - mean_2))

within_class_scatter = np.add(wcs_1,wcs_2)

bcs_1 = np.multiply(len(df3_1),np.outer((m_1 - mean_all),(m_1 - mean_all)))
bcs_2 = np.multiply(len(df3_2),np.outer((m_2 - mean_all),(m_2 - mean_all)))

between_class_scatter = np.add(bcs_1,bcs_2)

e_val, e_vector = np.linalg.eig(np.dot(lg.inv(within_class_scatter),between_class_scatter))
for e in range (len(e_val)):
    e_scatter = e_vector[:,e].reshape(2,1)

    print(e_val[e].real)

print(between_class_scatter)

total_lda = sum(e_val)
var_exp_lda = [(i / total_lda)*100 for i in sorted(e_val, reverse=True)]
var_exp_lda
cum_var_exp_lda = np.cumsum(var_exp_lda)
cum_var_exp_lda

eig_pairs = [(np.abs(e_val[i]).real, e_vector[:,i].real) for i in range(len(e_val))]


eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)



#print('Eigenvalues in decreasing order:\n')
#for i in eig_pairs:
 #   print(i[0])

W= eig_pairs[0][1].reshape(2,1)
lda_project = np.dot(df2,W)



plt.title('W plot')
plt.scatter(data_V1[0:30], data_V2[0:30], marker = 'o', color = 'purple', alpha = 0.5)
plt.scatter(data_V1[30:60],data_V2[30:60], marker = 'o', color = 'purple', alpha = 0.5)
plt.plot([0,100*myPCAResults['loadings'][0,0]], [0, 100*myPCAResults['loadings'][1,0]],
            color='orange', linewidth=3)
plt.xlim(0, 40), plt.ylim(0, 40)
plt.plot(60*W)

plt.show()



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('LDA')
ax.plot(lda_project[0:30], np.zeros(30), linestyle='None', marker='o', color='olive')
ax.plot(lda_project[30:60], np.zeros(30), linestyle='None', marker='o', color='purple')
ax.scatter(data_V1[0:30], data_V2[0:30], marker = 'o', color = 'purple', alpha = 0.5)
ax.scatter(data_V1[30:60],data_V2[30:60], marker = 'o', color = 'purple', alpha = 0.5)
fig.show()


# In[15]:

myPCAResults = d_PCA(data)
sqr_PCAdata= myPCAResults['scores']*myPCAResults['scores']
variance= np.sum(sqr_PCAdata,axis=0)/sqr_PCAdata.shape[0]-1
print('PCA Variance is :',variance)
total_varinace=np.sum(variance)
print('The total PCA varinace is :',total_varinace)
percentVarianceExplained = 100 * myPCAResults['PC_variance'][0] / sum(myPCAResults['PC_variance'])
print "PC1 explains: " + str(round(percentVarianceExplained, 2)) + '% variance\n'
percentVarianceExplained1 = 100 * myPCAResults['PC_variance'][1] / sum(myPCAResults['PC_variance'])
print "PC2 explains: " + str(round(percentVarianceExplained1, 2)) + '% variance\n'


# In[36]:

covariance_pc1pc2 = np.cov(myPCAResults['scores'][:,0],myPCAResults['scores'][:,1])
print(covariance_pc1pc2) 


# In[37]:

PC1_variance= np.var(myPCAResults['scores'][:,0])


# In[38]:

print(PC1_variance)


# In[40]:

PC2_varinace = np.var(myPCAResults['scores'][:,1])
print(PC2_varinace)


# In[54]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('scree plot')
ax.scatter(myPCAResults['scores'][:,0], myPCAResults['scores'][:,1], color='blue')
fig.show()


# In[55]:

PCaxis =  myPCAResults['data'].dot(myPCAResults['loadings'][0])
plt.title('Regression Plot')
plt.scatter(PCaxis[0:30], PCaxis[0:30], marker = 'o', color = 'purple', alpha = 0.5)
plt.scatter(PCaxis[30:60],PCaxis[30:60], marker = 'o', color = 'purple', alpha = 0.5)
plt.show()


# In[ ]:



