from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# Duomenu nuskaitymas is failo
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

# Nubraizomas demiu grafikas
def plot_data (data):
    fig, ax = plt.subplots()
    ax.plot(data[0], data[1])

    ax.set(xlabel='Metai', ylabel='Saulės dėmių skaičius',
       title='Saulės dėmių skaičius duotais metais')

    plt.show()

# Ivesties ir isvesties sarasai. Autoregresinio modelio eile = 2
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

# Trimatis grafikas
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

# Tikru ir prognozuojamu reiksmiu palyginimas
def plot_comparison (x, A, B):
    fig, ax = plt.subplots()

    ax.plot(x, A, label='Tikros reikšmės')
    ax.plot(x, B, label='Prognozuojamos reikšmės')
    ax.set(xlabel = 'Metai', ylabel='Saulės dėmių skaičius')
    ax.legend()

    plt.show()

# Prognozes klaidos vektoriaus ieskojimas
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

# MSE
def mse (n, err):
    err_sum = 0
    for i in err:
        err_sum += i*i
    mse = 1 / n * err_sum

    return mse

# MAD
def mad (err):
    median = np.median(np.absolute(err))
    return median

data = read()
plot_data(data)
pt = split(data, 10)
plot_io_data(pt)

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

plot_comparison(data[0][:200], Tu, Tsu)

# full data
Pu = pt[0]
Tu = pt[1]

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

plot_comparison(data[0][:len(Tu)], Tu, Tsu)

error = plot_error(Tsu,Tu,data[0][:len(Tsu)])

MSE = mse(len(error), error)
print("MSE=",MSE)

MAD = mad(error)
print("MAD=",MAD)
