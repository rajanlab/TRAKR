# -*- coding: utf-8 -*-
#############################################################
## Classifying MNIST digits using TRAKR
###########################################################################



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
import tensorflow as tf

##################################################
## extract all the ten digits
#################################################


(X_arr, y_arr), (
     Xtest,
     ytest,
 ) = tf.keras.datasets.mnist.load_data()

X=np.zeros((10,100,28,28))
y=np.zeros((10,100))
for i in range(10):
       tempx = X_arr[np.where((y_arr == i ))]
       tempy= y_arr[np.where((y_arr == i ))]
       ind = np.random.choice(np.size(tempx,0), size=100, replace=False)
       X[i,:,:,:]=tempx[ind,:]
       y[i,:]=tempy[ind]

X=X.reshape(1000,784)
y=y.reshape(1000)

##################################################
##  train on one digit
## and test on all other digits [and repeat]
##
## to try to classify into different digits later

## with an additional noise loop too, that runs same code multiple times on different noise levels
######################################################

# load prior saved data
#X=np.load('mnist_trakr_X_alldigits.npy')
#y=np.load('mnist_trakr_labels_alldigits.npy')

level=np.linspace(0,1,50) #testing 50 noise levels
totaltime=784
linewidth=4
svm = SVC()
svm_prob = SVC(probability=True)
scores=[]
aucvec=[]

for loop in range(len(level)):
#    X=np.load('mnist_trakr_X_alldigits.npy')
#    y=np.load('mnist_trakr_labels_alldigits.npy')
    sigma=level[loop]*255
    noise=np.random.normal(0,sigma,size=(1000,784))
    Xnoisy=X+noise
    learning_error_tot=np.zeros((np.size(X,0),np.size(X,0),totaltime))
    # loop for training on one digit, testing on all others
    for i in range(np.size(X,0)): 
        N_out=1
        N=30 # number of neurons in the RNN
        g=1.2 # gain
        tau=1 # tau
        delta = .3 # delta for Euler's method
        alpha=1 # alpha for regularizer
        regP=alpha*np.identity(N) # regularizer
        J = g*np.random.randn(N,N)/np.sqrt(N) # connectivity matrix J
        r = np.zeros((N, totaltime)) # rate matrix - firing rates of neurons
        x = np.random.randn(N, 1) # activity matrix before activation function applied
        z_out = np.zeros((N_out,totaltime)) # z(t) for the output read out unit
        error = np.zeros((N_out, totaltime)) # error signal- z(t)-f(t)
        learning_error = np.zeros((N_out, totaltime)) # change in the learning error over time
        w_out = np.random.randn(N, N_out)/np.sqrt(N) # output weights for the read out unit
        w_in = np.random.randn(N, N_out) # input weights 
        f=Xnoisy[i,:].reshape(1,-1)/np.max(Xnoisy[i,:])
        error,learning_error,z_out,w_out,x,regP,r=dynamics(N_out,N,g,tau,delta,f,
                        totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                        freezew=0,t1_train=0,
                        t2_train=totaltime)
        print(i)
        #testing all digits
        for j in range(np.size(X,0)):
            f=X[j,:].reshape(1,-1)/np.max(X[j,:])
            error,learning_error,z_out,w_out,_,_,r=dynamics(N_out,N,g,tau,delta,f,
                        totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                        freezew=1,t1_train=0,
                        t2_train=totaltime)
            learning_error_tot[i,j,:]=learning_error #matrix of errors (E(t))
            print(j)
    X=learning_error_tot #feeding errors into SVM classifier
    y_bin = label_binarize(y, classes=[0, 1, 2,3,4,5,6,7,8,9])
    n_classes = y_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros((n_classes))
    for k in range(5):
        # doing k-fold cv for the purposes of this code
        scores.append(cross_val_score(
                svm, X[k,:,:], y, cv=5, scoring ='accuracy'))
        y_score = cross_val_predict(svm_prob, X[k,:,:], y, cv=5 ,method='predict_proba')
        for l in range(n_classes):
            fpr[l], tpr[l], _ = roc_curve(y_bin[:, l], y_score[:, l])
            roc_auc[l] = auc(fpr[l], tpr[l])
        aucvec.append(roc_auc)
#         print(k)
    print(loop)