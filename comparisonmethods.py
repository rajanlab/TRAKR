########################################################################################
## Euclidean distance, DTW, Naive Bayes, Mutual Info based classification on seq-MNIST Dataset
##########################################################################################

## Uncomment each part to use a particular method
## doing k fold cv in the code here
#%%

#######################################################
## get euclidean distances between mnist digits 
## 
## to compare with trakr (error) based classification
########################################################

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

## get prior stored digits - can also generate new as shown in trakr_seqmnist.py
X=np.load('mnist_trakr_X_alldigits.npy')
y=np.load('mnist_trakr_labels_alldigits.npy')
X=X.reshape(-1,784)

#adding noise
sigma=.1*255
noise=np.random.normal(0,sigma,size=(1000,784))
X=X+noise
distmat=[]


svm = SVC()
svm_prob = SVC( probability=True)
scores=[]
y_bin = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9])
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = np.zeros((n_classes))
aucvec=np.zeros((np.size(X,0),n_classes))

for k in range(np.size(X,0)):
    distmat=[]
    for i in range(np.size(X,0)):
        distmat.append(np.linalg.norm(X[i,:] - X[k,:]))
    scores.append(cross_val_score(
        svm, np.array(distmat).reshape(-1,1), y, cv=5, scoring ='accuracy').mean())
    y_score = cross_val_predict(svm_prob, 
        np.array(distmat).reshape(-1,1), y, cv=5 ,method='predict_proba')
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    aucvec[k,:]=roc_auc
    print(k)


#%% DTW based classification
    
#######################################################
## DTW (dynamic time warping distances) between mnist digits
## to compare with trakr (learning error) based classification
########################################################

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

## get prior stored digits - can also generate new as shown in trakr_seqmnist.py
X=np.load('mnist_trakr_X_alldigits.npy')
y=np.load('mnist_trakr_labels_alldigits.npy')
X=X.reshape(-1,784)
distmat=[]

knn = KNeighborsClassifier(n_neighbors=1)
scores=[]
y_bin = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9])
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = np.zeros((n_classes))
aucvec=np.zeros((np.size(X,0),n_classes))

for k in range(np.size(X,0)):
    distmat=[]
    for i in range(np.size(X,0)):
        dist,_=np.array(fastdtw(X[i,:], X[k,:], dist=euclidean))
        distmat.append(dist)
    scores.append(cross_val_score(
        knn, np.array(distmat).reshape(-1,1), 
        y, cv=5, scoring ='accuracy').mean())
    y_score = cross_val_predict(knn, np.array(distmat).reshape(-1,1), 
                y, cv=5 ,method='predict_proba')
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    aucvec[k,:]=roc_auc
    print(k)

#%% Naive Bayes based classification

## Naive bayes - using for classification into mnist digits
## compare against trakr
##
##############################################################

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


## get prior stored digits - can also generate new as shown in trakr_seqmnist.py

X=np.load('mnist_trakr_X_alldigits.npy')
y=np.load('mnist_trakr_labels_alldigits.npy')
X=X.reshape(-1,784)
gnb = GaussianNB()
scores=[]
y_bin = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9])
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = np.zeros((n_classes))

scores.append(cross_val_score(
        gnb, X, y, cv=5, scoring ='accuracy'))
y_score = cross_val_predict(gnb, X, y, cv=5 ,method='predict_proba')
for j in range(n_classes):
    fpr[j], tpr[j], _ = roc_curve(y_bin[:, j], y_score[:, j])
    roc_auc[j] = auc(fpr[j], tpr[j])
auc=roc_auc

print(scores)
print(np.mean(auc))


#%% Mutual Information based classification

## Mutual Information - using for classification into mnist digits
## compare against trakr
##
##############################################################

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pyinform as pyinf


## get prior stored digits - can also generate new as shown in trakr_seqmnist.py


X=np.load('mnist_trakr_X_alldigits.npy')
y=np.load('mnist_trakr_labels_alldigits.npy')
X=X.reshape(-1,784)
scores=[]
y_bin = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9])
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = np.zeros((n_classes))
svm = SVC()
svm_prob = SVC( probability=True)
aucvec=np.zeros((np.size(X,0),n_classes))

for k in range(np.size(X,0)):
    minfo=[]
    for i in range(np.size(X,0)):
        minfo.append(pyinf.mutual_info(X[i,:], X[k,:]))
    scores.append(cross_val_score(
        svm, np.array(minfo).reshape(-1,1), 
        y, cv=5, scoring ='accuracy'))
    y_score = cross_val_predict(svm_prob, np.array(minfo).reshape(-1,1), 
                y, cv=5 ,method='predict_proba')
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    aucvec[k,:]=roc_auc
    print(k)






















