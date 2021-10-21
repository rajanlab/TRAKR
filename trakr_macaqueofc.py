## Macaque OFC Data - TRAKR

##################################################
##  train on one trial (rest)
## and test on all other trials in a day [and repeat]
##
## to try to classify into different behavioral epochs later
######################################################

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
# data=filtered_data # raw data available for download upon request

totaltime=6436
time = np.linspace(0,totaltime/1e3,totaltime).reshape(1,totaltime) # time vector

linewidth=3
 
bhv=bhv.dropna()

print(np.size(bhv,0))

z_out_tot=np.zeros((np.size(bhv,0),np.size(bhv,0),totaltime))
learning_error_tot=np.zeros((np.size(bhv,0),np.size(bhv,0),totaltime))
f_tot=np.zeros((np.size(bhv,0),np.size(bhv,0),totaltime))
error_tot=np.zeros((np.size(bhv,0),np.size(bhv,0),totaltime))

# loop for training one one, testing on all others
for i in range(np.size(bhv,0)): 
    # hyperparameters
    N_out=1
    N=30 # number of neurons in the RNN
    g=1.2 # gain
    tau=1 # tau
    delta = .1 # delta for Euler's method
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
#     f=np.mean(data[:,np.int(bhv['FixPtOn'].iloc[i]) : 
#                     np.int(bhv['FixPtOn'].iloc[i])+totaltime],axis=0).reshape(1,-1)
    f=data[:,np.int(bhv['FixPtOn'].iloc[i]) : 
                    np.int(bhv['FixPtOn'].iloc[i])+totaltime]
        #training
    error,learning_error,z_out,w_out,x,regP,r=dynamics(N_out,N,g,tau,delta,f,
                    totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                    freezew=0,t1_train=np.int(bhv['FirstJuiceOn'].iloc[i])-np.int(bhv['FixPtOn'].iloc[i]),
                    t2_train=np.int(bhv['RespCueOFFtime'].iloc[i])-np.int(bhv['FixPtOn'].iloc[i]))
#     print()
    print(i)
#     print()
    for j in range(np.size(bhv,0)):
#         f=np.mean(data[:,np.int(bhv['FixPtOn'].iloc[j]) : 
#                     np.int(bhv['FixPtOn'].iloc[j])+totaltime],axis=0).reshape(1,-1)
        #testing
        f=data[:,np.int(bhv['FixPtOn'].iloc[j]) : 
                    np.int(bhv['FixPtOn'].iloc[j])+totaltime]
        error,learning_error,z_out,w_out,_,_,r=dynamics(N_out,N,g,tau,delta,f,
                    totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                    freezew=1,t1_train=0,
                    t2_train=totaltime)
        z_out_tot[i,j,:]=z_out
        learning_error_tot[i,j,:]=learning_error
        f_tot[i,j,:]=f
        error_tot[i,j,:]=error
#         print(j)


#%% classification

lerror_data=learning_error_tot
# lerror_data=np.load('chip060519_trainononetrialbaseline_lerror_testonallothers.npy')

bhv=bhv.dropna()
count=500
scores=[]
svm = SVC()
svm_prob = SVC(probability=True)
features=[]
y=np.concatenate((np.ones((np.size(lerror_data,0))),2*np.ones((np.size(lerror_data,0))),
                      3*np.ones((np.size(lerror_data,0)))))
y_bin = label_binarize(y, classes=[1, 2, 3])
n_classes = y_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = np.zeros((n_classes))
aucvec=np.zeros((np.size(lerror_data,0),n_classes))

for k in range(1):
    vec1=[]
    vec2=[]
    vec3=[]
    for i in range(np.size(lerror_data,0)):
        vec1.append(lerror_data[k,i,np.int(bhv['FixPtOff'].iloc[i])-np.int(bhv['FixPtOn'].iloc[i]):
                      np.int(bhv['FixPtOff'].iloc[i])-np.int(bhv['FixPtOn'].iloc[i])+count])
    for i in range(np.size(lerror_data,0)):
        vec2.append(lerror_data[k,i,np.int(bhv['ValueCueONtime'].iloc[i])-np.int(bhv['FixPtOn'].iloc[i]):
                  np.int(bhv['ValueCueONtime'].iloc[i])-np.int(bhv['FixPtOn'].iloc[i])+count])
    for i in range(np.size(lerror_data,0)):
        vec3.append(lerror_data[k,i,np.int(bhv['RespCueONtime'].iloc[i])-np.int(bhv['FixPtOn'].iloc[i]):
                  np.int(bhv['RespCueONtime'].iloc[i])-np.int(bhv['FixPtOn'].iloc[i])+count])
    X=np.concatenate((vec1,vec2,vec3))
#    X_new = SelectKBest(chi2, k=numfeat).fit_transform(X, y)
#    features.append(SelectKBest(chi2, k=numfeat).fit(X, y).get_support(indices=True)) #list of booleans
    #k-fold cv for the purposes of this code
    scores.append(cross_val_score(
            svm, X, y, cv=5, scoring ='accuracy').mean())
    y_score = cross_val_predict(svm_prob, X, y, cv=5 ,method='predict_proba')
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    aucvec[k,:]=roc_auc
    print(k)
    
print(roc_auc)

#%% plotting
# plt.figure()
# plt.hist(scores)
# plt.xlabel('Accuracy',color='white')
# plt.ylabel('#trials',color='white')
# plt.tick_params(colors='white')
# plt.xlim([0,1])

# plt.figure()
# plt.hist(np.mean(aucvec,axis=1))
# plt.xlabel('AUC',color='white')
# plt.ylabel('#trials',color='white')
# plt.tick_params(colors='white')
# plt.xlim([0,1])

