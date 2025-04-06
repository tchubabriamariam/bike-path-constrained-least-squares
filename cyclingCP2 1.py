import matplotlib.pyplot as plt
import numpy as np
X = [1, 2, 3, 6, 9, 11, 15, 18,11,12,1,20]
Y = [4, 3, 1, 6, 6, 9, 15, 10,12,13,4,21]




plt.scatter(X, Y, color='red', label='Bumpy Points')
plt.scatter([], [], color='Black', label='Inverse')
plt.scatter([], [], color='Blue', label='CGS')
plt.scatter([], [], color='purple', label='MGS')
plt.scatter(12, 9, color='green', label='Constraint Point (12, 9)', marker='X', s=100)
plt.title('Road Description')
plt.xlabel('Horizontal Distance')
plt.ylabel('Elevation')
plt.legend()
plt.grid(True)
# plt.show()

A=[]
def Bike_A(A, X):
    for i in range(len(X)):
     A.append([1, X[i]])
    return np.array(A)
A=Bike_A(A,X)
b=[]

b=np.array([Y])
b=b.T



## Ac=b
def transpose(A):
    T = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            T[j][i]=A[i][j]
    return np.array(T)

##Cc=d
(12,9)
C=np.array([[1,12]])
d=np.array([[9]])
A_T=transpose(A)
C_T=C.T
A_T_A=np.dot(A_T,A)

KKT = np.block([
    [2 * A_T_A, C_T],
    [C, np.array([[0]])]
])
KKT_inverse=np.linalg.inv(KKT)
print(KKT_inverse)

def right_side(A_T,b,d):
    A_T_b=np.dot(A_T,b)
    res=[]
    for i in range(len(A_T_b)):
        res.append(2*A_T_b[i])
    for i in range(len(d)):
        res.append(d[i])
    # res = np.concatenate((A_T_b, d))
    return np.array(res)

r=right_side(A_T,b,d)        
sol=np.dot(KKT_inverse,r)

c=[]
def last_step(sol,c):
    c=[]
    for i in range(len(A[0])):
        c.append(sol[i])
    return c 

c=last_step(sol,c)

def f(x,c):
    f=0
    for i in range(len(c)):
        f+=c[i]*x**i
    return f


x_values = np.linspace(min(X), max(X), 100)
y_values = [f(x, c) for x in x_values]
plt.plot(x_values, y_values, color='Black', label=f'Line of Best Fit: y = {c[1][0]:.2f}x + {c[0][0]:.2f}')
# plt.show()   

#######################!!!!!!!!!!!!!!!!!!!!!!   3.2  !!!!!!!!!!!!!!!!!!!!!##########################
R=[]
Q=[]
def extract_column(A, i):
    if len(A[0])<=i:
        raise Exception
    return [row[i] for row in A]

def norm_2(x):
    return np.array(sum((x[i])**2 for i in range(len(x)))**(1/2))

def gram_schmidt_c(A):
    n=len(A)
    m=len(A[0])
    Q=np.zeros((n, m))
    U=[]
    R=np.zeros((m, m))
    def a(i):
        return extract_column(A,i)
    def e(i):
        return U[i]/norm_2(U[i])
    for i in range(m):
        if i==0:
            U.append(a(0))
        else:
            u=a(i)
            for j in range(i):
                u-=np.dot(a(i),e(j))*e(j)
            U.append(u)       
    for i in range(m):
        Q[:,i]=e(i)   
    for i in range(m):
        for j in range(i,m):
         R[i, j] = np.dot(e(i), a(j)) 
    return Q,R

n=len(A)
m=len(A[0])
k=len(C)
def new_matrix(A,C):
    M=[]
    for row in A:
        M.append(row)
    for row in C:
        M.append(row)
    return np.array(M) 
M=new_matrix(A,C)
Q,R=gram_schmidt_c(M)

def Q_s(Q):
    Q1=[]
    Q2=[]
    for i in range(n):
        Q1.append(Q[i])
    for i in range(n,n+k):
        Q2.append(Q[i])
    return np.array(Q1),np.array(Q2) 

Q1,Q2=Q_s(Q)
Q2_T=transpose(Q2)
Q_tilt,R_tilt=gram_schmidt_c(Q2_T) 
Q_tilt_T=transpose(Q_tilt)
Q1_T=transpose(Q1)
R_tilt_inverse=np.linalg.inv(R_tilt)
R_tilt_inverse_T=transpose(R_tilt_inverse)
R_inverse=np.linalg.inv(R)
print(A.shape)
print(Q.shape)
print(Q2.shape)
print(Q_tilt.shape)
print(Q_tilt_T.shape)
print(Q1_T.shape)
print(b.shape)
e=np.dot(R_tilt_inverse,(np.dot(Q_tilt_T,np.dot(Q1_T,b))-2*np.dot(R_tilt_inverse_T,d)))
y=0.5*(np.dot(Q1_T,b)-np.dot(Q2_T,e))
c1=[]
c1=np.dot(R_inverse,y)

x_values = np.linspace(min(X), max(X), 100)
y_values = [f(x, c1) for x in x_values]
plt.plot(x_values, y_values, color='blue', label=f'Line of Best Fit: y = {c1[1][0]:.2f}x + {c1[0][0]:.2f}')
# plt.show()
#######################!!!!!!!!!!!!!!!!!!!!!!   3.3  !!!!!!!!!!!!!!!!!!!!!##########################

def projection(x,y):
    x_y=np.dot(x,y)
    x_x=np.dot(x,x)
    return (x_y/x_x)*x

def gram_schmidt_m(A):
    n=len(A)
    m=len(A[0])
    tmp=[]
    Q=np.zeros((n, m))
    R=np.zeros((m, m))
    def a(i):
        return extract_column(A,i)
    for i in range(m):
        tmp.append(a(i))
    for i in range(m):       
              
        for j in range(i):
             tmp[i]=tmp[i]-projection(Q[:,j],tmp[i])
        Q[:,i]=tmp[i]/norm_2(tmp[i]) 
    Q_T=transpose(Q)
    R=np.dot(Q_T,A)
    return Q,R


n=len(A)
m=len(A[0])
k=len(C)
def new_matrix(A,C):
    M=[]
    for row in A:
        M.append(row)
    for row in C:
        M.append(row)
    return np.array(M) 
M=new_matrix(A,C)
Q,R=gram_schmidt_m(M)

def Q_s(Q):
    Q1=[]
    Q2=[]
    for i in range(n):
        Q1.append(Q[i])
    for i in range(n,n+k):
        Q2.append(Q[i])
    return np.array(Q1),np.array(Q2) 

Q1,Q2=Q_s(Q)
Q2_T=transpose(Q2)
Q_tilt,R_tilt=gram_schmidt_c(Q2_T) 


Q_tilt_T=transpose(Q_tilt)
Q1_T=transpose(Q1)

R_tilt_inverse=np.linalg.inv(R_tilt)
R_tilt_inverse_T=transpose(R_tilt_inverse)
R_inverse=np.linalg.inv(R)
print(A.shape)
print(Q.shape)
print(Q2.shape)
print(Q_tilt.shape)
print(Q_tilt_T.shape)
print(Q1_T.shape)
print(b.shape)
e=np.dot(R_tilt_inverse,(np.dot(Q_tilt_T,np.dot(Q1_T,b))-2*np.dot(R_tilt_inverse_T,d)))
y=0.5*(np.dot(Q1_T,b)-np.dot(Q2_T,e))
c2=[]
c2=np.dot(R_inverse,y)

x_values = np.linspace(min(X), max(X), 100)
y_values = [f(x, c2) for x in x_values]
plt.plot(x_values, y_values, color='purple', label=f'Line of Best Fit: y = {c2[1][0]:.2f}x + {c2[0][0]:.2f}')
plt.show()
for item in c:
    print(item)
print(c1)
print(c2)
def calculate_error(X, Y, c):
    error = sum((Y[i] - f(X[i], c))**2 for i in range(len(X)))
    return error
error_c = calculate_error(X, Y, c)
error_c1 = calculate_error(X, Y, c1)
error_c2 = calculate_error(X, Y, c2)

print(f"Error for Normal Equation Method: {error_c}")
print(f"Error for Classical Gram-Schmidt: {error_c1}")
print(f"Error for Modified Gram-Schmidt: {error_c2}")

