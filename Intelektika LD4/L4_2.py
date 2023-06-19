import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sympy import Lambda
from tensorflow import keras
from keras import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from cProfile import label
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score, KFold



dataset = "emissions_dataset.csv"

# Nuskaityti duomenis
df = pd.read_csv(dataset, sep=";")
#df = pd.read_csv(dataset)

# Atspausdinti duomenu rinkinio duomenis
print(df)
print(" ")

# Atspausdinti kategorinius
for column in df.columns:
    if df[column].dtype == 'object':
        print(f"{column}: {len(df[column].unique())}")

# Atspausdinti skaitinius
numerical = df.describe().T

# Atspausdinti skaitinius (issamiau)
numerical.style.background_gradient(cmap='Oranges')
print(numerical)
print(" ")

# Atspausdinti kategorinius kintamuosius
categorial = df.describe(include="object").T
categorial.style.background_gradient(cmap='Oranges')
print(categorial)
print(" ")

# Atspausdinti trukstamu reiksmiu kieki kiekviename stulpelyje
df.isnull().sum()

# Pasalinti trukstamas reiksmes
df.dropna(axis=0, inplace=True)

print(" ")
df.info()

# Pasalina isskirtis naudojant median absolute deviation (MAD) metodu (Is interneto)
# def remove_outliers(data, vars, thereshold = 3.5):  # 3.5 yra default
#     data_clean = data.copy()
#     for var in vars: 
#         median = np.median(data_clean[var]) # Paskaiciuoja mediana dabartiniam kintamajam
#         mad = np.median(np.abs(data_clean[var] - median)) # Paskaiciuoja median absolute deviation (MAD) dabartiniam kintamajam
#         if mad == 0: # Jeigu MAD == 0, kintamasis yra konstanta ir jo negalime salinti
#             continue
#         mad_z_scores = np.abs((data_clean[var] - median) / mad) # Paskaiciuoja z-scores dabartiniam kintamajam
#         data_clean = data_clean[mad_z_scores < thereshold] # Isfiltruoja eilutes kurios turi isskirtis dabartiniam kintamajam
#     return data_clean  

# df_cleaned_data = remove_outliers(df, ["Fuel_Consumption_City", "Fuel_Consumption_Highway", "CO2_Emissions"]) # Pasalina isskirtis

X = df.drop("Engine_Size", axis=1)
y = df["Engine_Size"]

# Encode engine_fuel kintamaji kaip Integer
# le = LabelEncoder() 
# y = le.fit_transform(y)

# Duomenų rinkinio padalijimas į mokymo ir testavimo rinkinius (80% training and 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Kategorinių stulpelių sudarymas pasitelkiant LabelEncoder ir transformuojant duomenis.
le = LabelEncoder()
X_train = X_train.apply(le.fit_transform) 
X_test = X_test.apply(le.fit_transform)


model = Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_dim=X.shape[1]))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

#print_weights = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, shuffle=True)




# def createModel():
#     model = Sequential()
#     model.add(keras.layers.Dense(16, activation='relu', input_dim=X_train.shape[1]))
#     model.add(keras.layers.Dense(10, activation='relu'))
#     model.add(keras.layers.Dense(1, activation='sigmoid'))
#     model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

#     return model

# cv = KFold(n_splits=10, shuffle=True)
# model = KerasClassifier(build_fn=createModel, epochs=20, batch_size=10, verbose=1)
# cv_scores = cross_val_score(model, X, y, cv=cv)

# cv_scores = []

# for train_index, test_index in cv.split(X):
#     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#     model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=1)
#     cv_scores.append(model.evaluate(X_test, y_test, verbose=1))


#print(f"Vidutinis tikslumo ivertis: {cv_scores.mean():.2f}") 






prediction = model.predict(X_test)

plt.plot(history.history['mse'])
plt.xlabel('Iter')
plt.ylabel('MSE')
plt.show()

# plt.plot(history.history['val_mse'])
# plt.xlabel('Iter')
# plt.ylabel('MSE')
# plt.show()