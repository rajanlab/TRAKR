#%% TRAKR functions
#%% TRAKR training and testing function
import numpy as np

#use this function to train a signal, or run a test signal through TRAKR

#inputs:

#N= number of neurons
#N_out= number of output neurons
#g=gain
#tau=integration time constant
#f=target function= I(t)
#totaltime=length of signal
#regP=regularizer
#J=recurrent connectivity matrix
#r=firing rates after nonlinearity
#x= activity before the nonlinearity (tanh)
#z_out= TRAKR output
#error;learning_error= E(t) initializations (from paper)
#w_out=output weights trained
#w_in=input weights
#freezew= a flag to freeze or unfreeze weights ; freezew=0 (unfreeze) ; freezew=1 (freeze)
#t1_train= starting point of the signal to train
#t2_train=ending point of the signal to train

def dynamics(N_out,N,g,tau,delta,f,totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,freezew,t1_train,t2_train):
    # for loop to train/test in real-time
    for t in range(totaltime):
            r[:, t] = np.tanh(x).reshape(N,) # activation function to calculate rates
            z_out[:,t] = np.dot(w_out.T,r[:,t].reshape(N,)) # zi(t)=sum (Jij rj) over j
            x = x + (-x + np.dot(J,r[:,t]).reshape(N,1) + np.dot(w_in,f[:,t]).reshape(N,1))*(delta/tau) # Euler update for activity x
            error[:,t] = z_out[:,t] - f[:,t] # z(t)-f(t)
            c=1/(1+ r[:,t].T@regP@r[:,t]) # learning rate
            regP = regP - c*(regP@r[:,t].reshape(N,1)@r[:,t].T.reshape(1,N)@regP) # calculating P(t)
            delta_w=c*error[:,t].reshape(N_out,1)*(regP@r[:,t]).T.reshape(1,N) # calculating deltaW for the readout unit
            learning_error[:,t] = np.sum(abs(delta_w),axis=1) # calculating error E(t)
            if freezew==0:
                if t>=t1_train and t <= t2_train:
                    w_out = w_out - delta_w.T # output weights being plastic
    return error,learning_error,z_out,w_out,x,regP,r











