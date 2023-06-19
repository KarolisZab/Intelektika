from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sympy import Lambda
from tensorflow import keras
from keras import *



def read():
    years= []
    spots= []
    result = []

    file = open('sunspot.txt','r')

    lines = file.readlines()

    for line in lines:
        current = line.split('\t')
        years.append(int(current[0]))
        spots.append(int(current[1]))
    
    file.close()

    result.append(years)
    result.append(spots)
    return result

def plot_data (data):
    fig, ax = plt.subplots()
    ax.plot(data[0], data[1])

    ax.set(xlabel='Metai', ylabel='Saulės dėmių skaičius',
       title='Saulės dėmių skaičius duotais metais')

    plt.show()

def split (data, n):
    P = [] #ivesties duomenys
    T = [] #isvesties duomenys

    spots = data[1]

    for i in range(len(spots)-n):
        temp = []
        for j in range(i, i + n):
            temp.append(spots[j])
        else:
            T.append(spots[j+1])
        P.append(temp)
    
    result = []
    result.append(P)
    result.append(T)

    return result

def plot_io_data (pt):
    P = pt[0]
    T = pt[1]

    P1 = []
    P2 = []

    ax = plt.axes(projection='3d')

    for i in range(len(P)-1):
        tuple = P[i]
        P1.append(tuple[0])
        P2.append(tuple[1])

    zdata = T[:-1]
    xdata = P1
    ydata = P2
    ax.scatter3D(xdata, ydata, zdata)
    ax.set(xlabel = 'Pirmos įvesties saulės dėmių skaičius', ylabel='Antros įvesties saulės dėmių skaičius', zlabel='Išvesties saulės dėmių skaičius', title='Įvesties ir išvesties sąrašų atvaizdavimas')

    plt.show()

def plot_comparison (x, A, B):
    fig, ax = plt.subplots()

    ax.plot(x, A, label='Tikros reikšmės')
    ax.plot(x, B, label='Prognozuojamos reikšmės')
    ax.set(xlabel = 'Metai', ylabel='Saulės dėmių skaičius')
    ax.legend()

    plt.show()

def plot_error (Ts, T, year):
    
    error = T-Ts

    plt.plot(year,error)
    plt.xlabel('Metai')
    plt.ylabel('Saulės dėmių skaičius')
    plt.legend()

    fig, ax = plt.subplots()
    ax.hist(error)
    ax.set(xlabel='Klaidos dydis', ylabel='Dažnis')
    plt.show()

    return error

def mean_sq_error (n, err):
    err_sum = 0
    for i in err:
        err_sum += i*i
    mse = 1 / n * err_sum

    return mse

def mean_abs_deviation (err):
    median = np.median(np.absolute(err))
    return median

data = read()
#plot_data(data)
pt = split(data, 2)
#plot_io_data(pt)

# data to 200 index -------
Pu = pt[0][:200]
Tu = pt[1][:200]

model= LinearRegression().fit(Pu, Tu)
w1 = model.coef_[0]
w2 = model.coef_[1]

b = model.intercept_

print('w1=', w1,' w2=', w2, ' b=', b)

r_sq = model.score(Pu,Tu)
print("R^2= ", r_sq)

print("intercept= ", model.intercept_)
print("slope= ", model.coef_ )

Tsu = model.predict(Pu)

#plot_comparison(data[0][:200], Tu, Tsu)

# full data
Pu = pt[0]
Tu = pt[1]

model = Sequential()
model.add(keras.layers.Dense(1,input_dim=2))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

print('Get_weights(): ', model.get_weights())

epochs = 1000

print_weights = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
history = model.fit(Pu,Tu, epochs=epochs, batch_size=10, callbacks=[print_weights])

Tsu = model.predict(Pu)

plt.plot(history.history['mse'])
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.show()



plot_comparison(data[0][:len(Tu)], Tu, Tsu)




