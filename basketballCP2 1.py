import matplotlib.pyplot as plt
import numpy as np
X = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,5,6,7,8,15,16,17]
Y = [4.2, 4.7, 5.1, 5.6, 6.0, 6.4, 5.8, 6.7, 6.9, 7.3, 7.0, 7.6, 8.1, 7.8, 8.5, 8.9, 9.1, 9.6, 9.8, 10.2,4.4,5.2,6,6.8,7.7,7.5,7.4]
plt.scatter(X, Y, color='red', label='Made Shots')
plt.scatter([], [], color='Black', label='Inverse')
plt.scatter([], [], color='Blue', label='CGS')
plt.scatter([], [], color='Purple', label='CGS')
plt.title('Kobe Bryant Made Shots')
plt.xlabel('Horizontal Distance (ft)')
plt.ylabel('Shot Height (ft)')
plt.legend()
plt.grid(True)


######################!!!!!!!!!!!!!!!!!!! SOLVE  Ac=b   !!!!!!!!!!!!!!!!!!!!!!!!!!!###############################
A=[]
def koby_A(A, X):
    for i in range(len(X)):
     A.append([1, X[i]])
    return A
     
A = koby_A(A, X)
print(A)

b=[]
b=Y
print(b)

# def koby_b(b,Y):
#     for i in range(len(X)):
#         b.append(Y[i])
#     return b

# C = [[1, 0], [0, 1]]
# d=[5,10]

def transpose(A):
    T = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            T[j][i]=A[i][j]
    return T

A_T=transpose(A)
# C_T=transpose(C)
# print(A_T)
##############################################!!!!!!!!!!!!!!!!!  3.1  !!!!!!!!!!!!########################
A_T_A=np.dot(A_T,A)
A_T_A_inverse=np.linalg.inv(A_T_A)
c1=np.dot(A_T_A_inverse,np.dot(A_T,b))
print(c1)

def f(x,c):
    f=0
    for i in range(len(c)):
        f+=c[i]*x**i
    return f
print(f(2,[1,2]))
    





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

Q,R=gram_schmidt_c(A)
R_inverse=np.linalg.inv(R) 
Q_T=transpose(Q)
A_dagger=np.dot(R_inverse,Q_T)
c2=np.dot(A_dagger,b)

print(np.array(A).shape)
print(Q.shape) 



      
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
    
         
Q,R=gram_schmidt_m(A)
R_inverse=np.linalg.inv(R) 
Q_T=transpose(Q)
A_dagger=np.dot(R_inverse,Q_T)
c3=np.dot(A_dagger,b)

x_values = np.linspace(min(X), max(X), 100)
y_values = [f(x, c1) for x in x_values]
plt.plot(x_values, y_values, color='Black', label=f'Line of Best Fit: y = {c1[1]:.2f}x + {c1[0]:.2f}')
# plt.legend()
x_values = np.linspace(min(X), max(X), 100)
y_values = [f(x, c2) for x in x_values]
plt.plot(x_values, y_values, color='blue', label=f'Line of Best Fit: y = {c2[1]:.2f}x + {c2[0]:.2f}')
# plt.legend()
x_values = np.linspace(min(X), max(X), 100)
y_values = [f(x, c3) for x in x_values]
plt.plot(x_values, y_values, color='purple', label=f'Line of Best Fit: y = {c3[1]:.2f}x + {c3[0]:.2f}')
# plt.legend()
plt.show() 
print(c1) 
print(c2)
print(c3) 
def calculate_error(X, Y, c):
    error = sum((Y[i] - f(X[i], c))**2 for i in range(len(X)))
    return error
error_c1 = calculate_error(X, Y, c1)
error_c2 = calculate_error(X, Y, c2)
error_c3 = calculate_error(X, Y, c3)

print(f"Error for Normal Equation Method: {error_c1}")
print(f"Error for Classical Gram-Schmidt: {error_c2}")
print(f"Error for Modified Gram-Schmidt: {error_c3}")
 
    
    
