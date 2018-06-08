# IMPLEMENTATION OF LINEAR REGRESSION FROM SCRATCH ONLY USING NUMPY 

#importing libraries

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#loading data
data= np.genfromtxt('diabetes-data.csv',delimiter=',');
features = ['age', 'sex', 'body mass index', 'blood pressure', 'serum1', 'serum2', 'serum3', 'serum4', 'serum5', 'serum6'];

#counting number of rows and column in data
(m,n)=data.shape;

#y_data is target data
#which we are going to predict on unseen values of features (test data in this case)
y_data=data[:,10:11];

#X_data is data corresponding to features of problem
X_data=data[:,0:10];




#defining function for plotting graph using matplotlib library
#graphs make us easy to visualize variation of estimated value correponding to known values
#three arguments X1-feature data, y-target data, y_pr-predicted data

def plotdata(X1,y,y_pr): 
    
    #plotting given data to visualize its nature correponding to single feature
    plt.plot(X1,y,'g^',label='validation data values');
    plt.xlabel('X');
    plt.ylabel('y');
    
    #if predicted value is not computed yet
    if (y_pr==np.zeros(y.shape, dtype = int)).all():
        plt.title('Plot Showing distributed data');
    #if predicted value is caculated
    else:
        #plotting predicted data on the above plot
        plt.plot(X1,y_pr,'ko',label='predicted values');
        plt.title('Plot Showing regression curve');
        
    #legend is a function in matplotlib module for describing the plotted data
    plt.legend();
    plt.show();
    
    
    

#vectorized implementation of costFunction and gradient        
#function to calculate average square error called cost function 
#and gradient that calculate change in cost function corresponding to small change in theta
#theta is initial value of theta that can be provided randomly
#Lambda is regularization coefficient , greater lambda lesser the affect of features on cost fuction(high bias)

def CostFunction(X,y,theta,Lambda):
    
    (m,n)=X.shape;
    
    #cost J= 1/2m*sum((X*theta-y)^2) + Lambda/2m*sum(theta^2)
    #NOTE- theta0 is not considered while regularization
    J= 1/(2*m) * sum((X.dot(theta) -y)**2) + Lambda/(2*m)* sum((theta[1:n])**2);
    
    #gradient = 1/m*sum((X'*theta-y)^2) + Lambda/m*theta
    #NOTE- theta0 is replaced by zero in regularization term
    theta_grad= (1/m)*((X.transpose()).dot((X.dot(theta)-y)))+ Lambda/m*(np.append(np.zeros([1,1]),theta[1:n],0)) ;
    
    return (J, theta_grad);




#function that normalize the provided data
#if the range of values of feature are significantly different then the visualization of computing will be ambigous
#features are first subracted by their mean then diveded by their standard deviation
def featureNormalization(X):
    
    #mean is numpy function that calculate average and axis (0 for columnwise and 1 for rowwise comutation)
    mu= np.mean(X,axis=0);
    
    #std calculate standard deviation of data
    sigma= X.std(0);
    
    #normalization step
    X= (X-mu)/sigma;
    
    return X;



#minimizing cost function using partial diffentiation (gradient of cost function - theta_grad)
#alpha is a constant that is choosen manually 
def GradientDescent(theta,alpha,X,y,Lambda):
    
    J=0;
    
    X= np.append(np.ones([X.shape[0],1]),X,1);
    
    #theta is shifted by the value of alpha*cost_gradient to the value where cost in minimum after every iteration 
    n_iteration= 100;
    
    for i in range(n_iteration):
        (J,theta_grad)= CostFunction(X,y,theta,Lambda);
       
        #storing cost after every iteration for plotting a graph of cost vs iteration
        J_history[i]=J;
        theta= theta- alpha*theta_grad;
        
    return theta,J;




#function to divide data into three parts 
    
def DistributeData(X,y,x_max):
    
    #first- data for training model
    #60 %
    a_train= X[0:int(0.6*m),0:x_max];
    b_train= y[0:int(0.6*m),0:x_max];
    
    #second- data for validation and model selection 
    #next 20%
    a_val= X[int(0.6*m):int(0.8*m),0:x_max];
    b_val= y[int(0.6*m):int(0.8*m),0:x_max];
    
    #third- test data for comparing predicted value and calculate accuracy
    #next 2O%
    a_test= X[int(0.8*m):m,0:x_max];
    b_test= y[int(0.8*m):m,0:x_max];
    return a_train,b_train,a_val,b_val,a_test,b_test;



 
#function to select the model wth highest degree of x(single feature)
#model is selected according to the minimum value of cost function corresonding to that model

def ModelSelection(X,y,alpha,Lambda):
    
    #model upto degree four are compared
    for i in range(1,5):
        x= X[:,0:i];
        
        #data distribution 
        x_train,y_train,x_val,y_val,x_test,y_test= DistributeData(x,y,i);
        
        #computing number of rows in each part
        m_train= x_train.shape[0];    
        m_val= x_val.shape[0];    
        m_test= x_test.shape[0];
        
        #initiolising the value of theta - vector or column matrixof all ones
        theta=np.ones([i+1,1]);
        
        #computing optimal value of theta using gradient descent
        optimal_theta,cst= GradientDescent(theta, alpha, x_train,y_train,Lambda);
        
        #storing the cost at optimal value of theta at every iteration
        J_cost[i-1]=J_history[99];
        
        #adding a column of ones to x_val (validation data)
        x_valm= np.append(np.ones([m_val,1]),x_val,1);
        
        #predicting value using optimal value of theta
        #prediction = theta*x (y= theta0 + theta1*x1 + theta2*x2 +.....+thetai*xi)
        y_predict= x_valm.dot(optimal_theta);
        
        #plotting predicted value corresponding to every feature value
        plotdata(x_val[:,0],y_val,y_predict);
        
        #adding a column of ones to x_test (test data)
        x_testm= np.append(np.ones([m_test,1]),x_test,1);
        
        #prediction corresponding to test data
        y_testpredict= x_testm.dot(optimal_theta);
        
        #calculation of error between predicted value and given data
        error= 100*((y_test- y_testpredict)).mean(0)/y_test.mean(0);
        print('percentage accuracy='+str(100-error));





X_data= featureNormalization(X_data);
y_data= featureNormalization(y_data);

#selection third feature- body mass index from X for training univariate model 
X= X_data[:,2:3];
y= y_data[:,0:1];

#distributing data into training, validation and test part    
a_train,b_train,a_val,b_val,a_test,b_test= DistributeData(X,y,1);

#computing number of rows
m_train= a_train.shape[0];    
m_val= a_val.shape[0];    
m_test= a_test.shape[0];

b_pr=np.zeros([m_test,1],int);

#visualizing nature of test data 
plotdata(a_test,b_test,b_pr);

#initialising value of lambda and alpha 
#NOTE- much greater value of alpha can lead to diversion of average square error(cost)
Lambda=0.01;
alpha=0.07;

#initialising theta to a column vector with two rows of all ones
theta=np.ones([2,1]);

#initialing vector to store cost at eace ith iteration 
J_history= np.zeros([100,1]);



print('Running UNIVARIATE REGULARISED LINEAR REGRESSIION\n')
optimal_theta,cst= GradientDescent(theta, alpha, a_train,b_train,Lambda);
a_testm= np.append(np.ones([m_test,1]),a_test,1);
b_pr= a_testm.dot(optimal_theta);
plotdata(a_test,b_test,b_pr);
error= 100*(abs(b_test.mean(0)- b_pr.mean(0)))/b_test.mean(0);
print('percentage accuracy='+str(100-error));



#plotting cost corresponding to each iteration of gradient descent vs number of iteration
#help to visualise number of iteration required to minimize cost

plt.plot(range(J_history.shape[0]),J_history,'b-',linewidth=3);
plt.xlabel('number of iteration');
plt.ylabel('cost')
plt.title('cost vs number of iteration');
plt.show();



print('Running SINGLE FEATURE LINEAR REGRESSION WITH DEGREE ONE AND GREATER THAN ONE');
print('\n\nModel selection with lowest cost')
Xmulti= X;

#adding column to X where at each iteration new column is ith power of x 
for i in  range(2,6):
    
    #append used to concatenate two matrix of same number of rows
    Xmulti= np.append(Xmulti,X**i,1);
    
J_cost= np.zeros([4,1]);

#running model selection function
ModelSelection(Xmulti,y,alpha,Lambda);

#argmin is a fuction in numpy library for calcualting minimum value in numpy array
#model with minimum cost is selected
selected_model_degree= np.argmin(J_cost);
print('----------------------------------------------------------');
print('\n\nhighest degree of model selected : '+str(selected_model_degree));
print('\n----------------------------------------------------------');




print('MULTIVARIATE GRADIENT DESCENT- taking all the features')
theta=np.ones([11,1]);

#Distribution of data into three parts (Training data, Validation data, Test data)-----------------------------------------------------------------
a_train_m,b_train_m,a_val_m,b_val_m,a_test_m,b_test_m= DistributeData(X_data,y_data,10);

#computing optimal value to minimize average square error using gradient descent.
optimal_theta_m,cst= GradientDescent(theta, alpha, a_train_m,b_train_m,Lambda);

#addition of an extra column containing ones 
#so that cofficient corresponding to constant can be calculated
a_test_m= np.append(np.ones([a_test_m.shape[0],1]),a_test_m,1);

#computing predicted value b_pr= theta0+x1*theta1+......+x10*theta10
#optimal_theta= [theta0, theta1.......,theta10] 
b_pr_m= a_test_m.dot(optimal_theta_m);

#calculation of error diffentiating mean of predicted value and observed value
error= 100*(abs(b_test_m.mean(0)- b_pr_m.mean(0)))/b_test_m.mean(0);
print('percentage accuracy='+str(100-error));