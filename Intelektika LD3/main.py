import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
import graphviz
import plotly.graph_objs as go


#dataset = "data/emissions_dataset.csv"
dataset = "data/cars_dataset.csv"

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

#  Patikrinimas isskirtims
def plot_comparison(data):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=data[var],orient= 'y',  ) 
    plt.title('Outliers') 
plt.show()

# # Atvaizduoti isskirtis
# for var in ['Fuel_Consumption_City', 'Fuel_Consumption_Highway', 'CO2_Emissions']:
#     plot_comparison(df)

# Atvaizduoti isskirtis
for var in ['odometer_value', 'price_usd', 'up_counter']:
    plot_comparison(df)

# Pasalina isskirtis naudojant median absolute deviation (MAD) metodu (Is interneto)
def remove_outliers(data, vars, thereshold = 3.5):  # 3.5 yra default
    data_clean = data.copy()
    for var in vars: 
        median = np.median(data_clean[var]) # Paskaiciuoja mediana dabartiniam kintamajam
        mad = np.median(np.abs(data_clean[var] - median)) # Paskaiciuoja median absolute deviation (MAD) dabartiniam kintamajam
        if mad == 0: # Jeigu MAD == 0, kintamasis yra konstanta ir jo negalime salinti
            continue
        mad_z_scores = np.abs((data_clean[var] - median) / mad) # Paskaiciuoja z-scores dabartiniam kintamajam
        data_clean = data_clean[mad_z_scores < thereshold] # Isfiltruoja eilutes kurios turi isskirtis dabartiniam kintamajam
    return data_clean  

#df_cleaned_data = remove_outliers(df, ["Fuel_Consumption_City", "Fuel_Consumption_Highway", "CO2_Emissions"]) # Pasalina isskirtis
df_cleaned_data = remove_outliers(df, ["odometer_value", "price_usd", "up_counter"]) # Pasalina isskirtis

# Atvaizduoja palyginima tarp pradiniu duomenu ir duomenu su pasalintomis isskirtimis
def plot_comparison(data, data_removed, var):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    sns.boxplot(y = data[var])
    plt.title('Original Data')

    plt.subplot(1, 4, 4)
    sns.boxplot(y = data_removed[var])
    plt.title('Removed')

    plt.suptitle(f'Comparison for {var}')
    plt.show()


# # Pasalinas isskirtis naudojant MAD metoda
# training_data = remove_outliers(df, ['Fuel_Consumption_City', 'Fuel_Consumption_Highway', 'CO2_Emissions'])

# # Plot comparison for each continuous variable
# for var in ['Fuel_Consumption_City', 'Fuel_Consumption_Highway', 'CO2_Emissions']:
#      plot_comparison(df, training_data, var)

# print(training_data['Fuel_Consumption_City'].min())


# Pasalinas isskirtis naudojant MAD metoda
training_data = remove_outliers(df, ['odometer_value', 'price_usd', 'up_counter'])

# Atvaizduoja palyginima kiekvienam isskirciu turinciam kintamajam
for var in ['odometer_value', 'price_usd', 'up_counter']:
    plot_comparison(df, training_data, var)

print(training_data['odometer_value'].min())


#df_cleaned_data['Fuel_Consumption_City'].min()
#df_cleaned_data['odometer_value'].min()

# --------------- 2. Prognozuojamas atributas -----------------

# Pasirinktas sprendimo medzio atributas - engine_fuel / Fuel_Type
# X = df_cleaned_data.drop("Fuel_Type", axis=1) # Drop the target variable Fuel Type
# y = df_cleaned_data["Fuel_Type"] # Target variable is Fuel Type

X = df_cleaned_data.drop("engine_fuel", axis=1)
y = df_cleaned_data["engine_fuel"]

# Encode engine_fuel kintamaji kaip Integer
le = LabelEncoder() 
y = le.fit_transform(y)

# ------------- 3-4. Duomenu rinkinio padalinimas -----------------

# Duomenų rinkinio padalijimas į mokymo ir testavimo rinkinius (80% training and 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# ---------------- 5. Sprendimų medžio sudarymas ------------------

# Kategorinių stulpelių sudarymas pasitelkiant LabelEncoder ir transformuojant duomenis.
le = LabelEncoder()
X_train = X_train.apply(le.fit_transform) 
X_test = X_test.apply(le.fit_transform)


# Sukuriamas Decision Tree modelis su Gini
clf= DecisionTreeClassifier(criterion="gini", random_state=42)
# Apsimokymas
clf.fit(X_train, y_train)


# --------------------- 6. Sprendimų medžio vizualizacija --------- 

unique_classes = np.unique(y) # Gauti unikalias reikšmes
str_unique_classes = [str(x) for x in unique_classes] # Konvertuoti unikalias klasifikatorių reikšmes į string tipo reikšmes
dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names = str_unique_classes , filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data)
graph.render("decision_tree2")


plt.figure(figsize =(20,10))
plot_tree(clf, filled= True, rounded= True, class_names= ['gasoline', 'diesel', 'gas', 'electric', 'hybrid-diesel', 'hybrid-petrol'], feature_names= X.columns)
plt.show()


# ---------------- 7. Sprendimų medžio testavimas ----------------

# Prognozuojame testavimo duomenis 
y_pred = clf.predict(X_test)

# Apskaiciuojmas  modelio tikslumas
accuracy = accuracy_score(y_test, y_pred)
print(" ")
print(f"Accuracy: {accuracy * 100 :.4f} ")
print(" ")

# Apskaiciuojamos paklaidos
mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred)
rse = r2_score (y_test, y_pred)
print(f"Mean Absolute Error: {mae: .4f}")
print(f"Mean Squared Error: {mse: .4f}")
print(f"Root Squared Error: {rse: .4f}")
print(" ")

# Atvaizduojama confusion matrica
conf_matrix = confusion_matrix(y_test, y_pred) # Apskaiciuojama confusion matrica
sns.heatmap(conf_matrix, annot=True, fmt="g")
plt.title("Confusion Matrix", fontsize=16)
plt.ylabel("Prediction", fontsize=14)
plt.xlabel("Actual",fontsize=14)
plt.show()


# ---------------- 8. Keičiamas gylis ------------------

for depth in range(3, 7):
    clf_depth = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=depth) # Sukuriamas Decision Tree modelis su Gini
    clf_depth.fit(X_train, y_train) # Apsimokymas
    y_pred = clf_depth.predict(X_test) # Prognozuojame testavimo duomenis
    accuracy = accuracy_score(y_test, y_pred) # Apskaiciuojmas  modelio tikslumas
    print(f"Depth accuracy {depth}: {accuracy * 100 :.4f} ") # Spausdinamas modelio tikslumas