import numpy as np
from sklearn.metrics import mean_squared_error

def step(z):
    if(z>=2+b):
        return 1
    else:
        return 0


def perceptronAND(input,w,b):
    z=w[0]*input[0]+w[1]*input[1]+b
    return z

w=[0,0]
print("Initial weights: ",w)
test=[[0,0],[0,1],[1,0],[1,1]]
y=[0,0,0,1]
t=1
b=1
p=[]


for i in range(4):
    flag=0
    while(flag==0):
        z=perceptronAND(test[i],w,b)
        z=step(z)
        #print("z,y=",z,y[i])
        #input()
        if(z!=y[i]):
            dw1=(t-z)*test[i][0]
            dw2=(t-z)*test[i][1]
            w[0]=w[0]+dw1
            w[1]=w[1]+dw2
            #print(w)
        else:
            flag=1

for i in range(0,4):
    y_pred=perceptronAND(test[i],w,b)
    y_pred=step(y_pred)
    p.append(y_pred)
    #print("ypred",y_pred)

print("Final weights=",w)
print("Testing inputs: ",test)
print("Actual outputs: ",y)
print("Predicted outputs: ",p)
print(mean_squared_error(p,y))