import numpy as np
from scipy.linalg import eigh
from scipy import optimize

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N, d = X.shape 
        M, _ = Y.shape
        X = X.reshape(N,1,d,1)
        Y = Y.reshape(1,M,d,1)
        G = ((X-Y).transpose((0, 1, 3, 2))@(X-Y)).reshape(N,M)
        return np.exp(-G/(2*self.sigma**2))
    
class Linear:
    def __init__(self): 
        self = self
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return X @ Y.T
    
class Polynomial:
    def __init__(self, d = 3, cst = 0):
        self.d = d  ## the degree of the polynomial
        self.cst = cst  ## trading parameter
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return np.power(X @ Y.T + self.cst,self.d)
    
class KernelPCA:
    def __init__(self, kernel, n_components = 10):
        self.kernel = kernel
        self.n_components = n_components
        
    def fit(self, X_train):
        
       #### You might define here any variable needed for the rest of the code  
        
        K = self.kernel(X_train,X_train)
        N = K.shape[0]
        
        # Center the gram matrix
        U = np.ones((N,N))/N
        Kc = (np.eye(N) - U)@K@(np.eye(N) - U)
        
        # Eigenvectors computations
        eigenvals, eigenvects = eigh(Kc)
        eigenvects = eigenvects.T
        
        eigenvals = eigenvals[::-1]   # sorting in descending order
        eigenvects = eigenvects[::-1]
        
        index = np.where(eigenvals>0) # keep only eigenvectors related to positive eigenvalues
        eigenvals = eigenvals[index]
        eigenvects = eigenvects[:,index].squeeze()
        
        # Normalization
        eigenvects = eigenvects / np.sqrt(eigenvals)
        
        # Projection 
        return np.dot(Kc,eigenvects)[:,:self.n_components]
    
class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.diag = None

    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K = self.kernel(X,X)
        y_diag = np.diag(y)
        
        # Lagrange dual problem
        def loss(alpha):
            #'''--------------dual loss ------------------ '''
            return - alpha.sum() + 0.5 * alpha.T @ y_diag @ K @ y_diag @ alpha

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            # '''----------------partial derivative of the dual loss wrt alpha-----------------'''
            return - np.ones(N) + y_diag @ K @ y_diag @ alpha


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha: (0 - y.T @ alpha).reshape(1,1) # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha:  - y #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
    
        fun_ineq = lambda alpha:  self.C*np.vstack((np.ones((N,1)),np.zeros((N,1)))) - (np.vstack((np.eye(N),-np.eye(N)))@alpha).reshape(2*N,1) # '''---------------function defining the ineequality constraint-------------------'''     
        jac_ineq = lambda alpha:   - np.vstack((np.eye(N),-np.eye(N))) # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
            
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})
        
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)

        self.alpha = optRes.x 
        ## Assign the required attributes
        
        # Support vectors on the margin
        supportIndices = np.logical_and(self.alpha>self.epsilon, self.alpha<self.C-self.epsilon)
        self.support = X[supportIndices] #'''------------------- A matrix with each row corresponding to a support vector ------------------'''
        
        self.b = (y - y_diag @ self.alpha @ K)[supportIndices].mean() #''' -----------------offset of the classifier------------------ '''
        self.norm_f = np.sqrt(self.alpha.T @ y_diag @ K @ y_diag @ self.alpha) # '''------------------------RKHS norm of the function f ------------------------------'''
        
        # Support vectors & intermediate variable 
        # for computing the separating function
        self.X_sp = X[np.where(self.alpha>self.epsilon)]
        self.diag = np.diag(y[np.where(self.alpha>self.epsilon)])@ self.alpha[np.where(self.alpha>self.epsilon)]

    ### Implementation of the separating function $f$
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.diag.T @ self.kernel(self.X_sp,x)
    
    
    def predict(self, X):
        """ Predict y values in {label1, label2} """
        d = self.separating_function(X)
        return 2*(d+self.b > 0) - 1
 


class KernelCCA:
    
    def __init__(self, kernel_a, kernel_b, tau_a = 0.5, tau_b = 0.5):
        self.type = 'non-linear'
        self.kernel_a = kernel_a  
        self.kernel_b = kernel_b    
        self.tau_a = tau_a  
        self.tau_b = tau_b   
        self.alpha = None
        self.beta = None
        self.support = None
        self.norm_f = None
        self.diag = None

    def fit(self, X, y):
            
        #### You might define here any variable needed for the rest of the code
        N = len(y)
        Ka = self.kernel_a(X,X)
        Kb = self.kernel_b(X,X)
        tau_a = self.tau_a
        tau_b = self.tau_b
        y_diag = np.diag(y)
            
        # Lagrange dual problem
        def loss(params):        #'''--------------dual loss ------------------ '''
            alpha, beta = params
            return - alpha.T @ Ka @ Kb @ beta

        fun_eq = lambda params: (1-(1-tau_a)*params[0].T@Ka@Ka@params[0] - tau_a*params[0].T@Ka@params[0],1-(1-tau_b)*params[1].T@Kb@Kb@params[1] - tau_b*params[1].T@Kb@params[1]) # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda params:  (-  2*(1-tau_a)*Ka@Ka@params[0] - 2*tau_a*Ka@params[0], -  2*(1-tau_b)*Kb@Kb@params[1] - 2*tau_b*Ka@params[1]) #'''----------------jacobian wrt alpha of the  equality constraint------------------'''


        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq})

        optRes = optimize.minimize(fun=lambda params: loss(params),
                                   x0= [np.ones(N),np.ones(N)], 
                                   method='SLSQP', 
                                       #jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)

        self.alpha, self.beta = optRes.x 