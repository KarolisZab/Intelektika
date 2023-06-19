import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE

# Classifier metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import seaborn as sns
from scipy import stats

dataset = "data/emissions_dataset.csv"
#dataset = "data/loan_default_prediction_dataset.csv"

def readDataset():
    vehicle_data = pd.read_csv(dataset, sep=";")
    x = vehicle_data.iloc[:, :-1].values
    y = vehicle_data.iloc[:, -1].values
    vehicle_data.head()
    return vehicle_data


def groupData(vehicle_data):
    numerical = ['Engine_Size', 'Cylinders', 'Fuel_Consumption_City', 'Fuel_Consumption_Highway', 'Fuel_Consumption_Comb', 'CO2_Emissions']
    numerical_idx = vehicle_data.columns.get_indexer(numerical)
    categorical = ['Make', 'Model', 'Vehicle_Class', 'Transmission', 'Fuel_Type']
    categorical_idx = vehicle_data.columns.get_indexer(categorical)
    return [categorical, numerical]


def createDataset():
    data = readDataset()
    df = pd.DataFrame(data)
    df = df.drop(df.columns[[0]], axis=1)
    variables = groupData(data)
    categorical = variables[0]
    numerical = variables[1]
    return df, categorical, numerical


def continuousAnalysis():
    global df_numerical
    print("Skaitinio tipo reiksmes")
    df_numerical = df.select_dtypes(['int64', 'float64'])
    print(df_numerical)
    print("\nBendras reiksmiu skaicius: ")
    print(df_numerical.count())
    print("\nTrukstamu reiksmiu skaicius:")
    print(df_numerical.isnull().sum())
    print("\nKardinalumas:")
    print(df_numerical.nunique())
    print("\nMinimali verte:")
    print(df_numerical.min())
    print("\nMaksimali verte:")
    print(df_numerical.max())
    # Atkerta apatinius 25% duomenu
    print("\nPirmas kvartilis:")
    print(df_numerical.quantile(.25))
    # Atkerta apatinius 75% duomenu
    print("\nTrecias kvartilis:")
    print(df_numerical.quantile(.75))
    print("\nVidurkis:")
    print(df_numerical.mean())
    # Mediana - pozymio reiksme, kuri dalina duomenis per puse
    # Vidurinė reikšmė, skirianti mažesnę ir didesnę imties puses	
    print("\nMediana:")
    print(df_numerical.median())
    # dydis, nusakantis atsitiktinio dydžio įgyjamų reikšmių sklaidą apie vidurkį 
    print("\nStandartinis nuokrypis:")
    print(df_numerical.std())


def categoricalAnalysis():
    global df_categorical, z
    print("Kategorinio tipo reiksmes")
    df_categorical = df.select_dtypes(include='object')
    print(df_categorical)
    print("\nBendras reiksmiu skaicius: ")
    print(df_categorical.count())
    print("\nTrukstamu reiksmiu skaicius:")
    print(df_categorical.isnull().sum())
    # Nusako, kiek unikaliu reiksmiu yra imtyje
    print("\nKardinalumas:")
    print(df_categorical.nunique())
    # Dazniausiai pasikartojanti pozymio reiksme imtyje
    print("\nModa:")
    print(df_categorical.mode())

    # Randamas pasikartojanciu daugiausiai daznumas
    print("\nModos daznumas:")
    # concatenate a series of boolean arrays, each of which indicates whether a value in a categorical DataFrame equals one of the mode values. 
    # The resulting concatenated array will have one row for each unique value in the DataFrame, and one column for each mode value. 
    # The sum of each column will give the count of values that match each mode.
    print(pd.concat([df_categorical.eq(x) for _, x in df_categorical.mode().iterrows()]).sum())

    print("\nModos procentinis daznumas:")
    # SecondMode Frequency proc = SecondMode frequency / count * 100
    print(pd.concat([df_categorical.eq(x) for _, x in df_categorical.mode().iterrows()]).sum() / len(
        df_categorical.index) * 100)
    
    print("\nAntroji moda:")
    temp_df = df_categorical
    modes = []
    for x, y in df_categorical.mode().iterrows():
        for z in range(0, len(y)):
            modes.append(y[z])
    for x in modes:
        temp_df = temp_df.replace(to_replace=x, value=np.nan, regex=True)
    print(temp_df.mode())

    # Isemus pirmos modos rastus duomenis, randama suma
    print("\nAntrosios modos daznumas:")
    print(pd.concat([df_categorical.eq(x) for _, x in temp_df.mode().iterrows()]).sum())
    
    # SecondMode Frequency proc = SecondMode frequency / count * 100
    print("\nAntrosios modos procentinis daznumas:")
    print(
        pd.concat([df_categorical.eq(x) for _, x in temp_df.mode().iterrows()]).sum() / len(df_categorical.index) * 100)


def Outliers():
    global z, df_numerical
    print("\nOutliers aptikimas:")
    z = np.abs(stats.zscore(df_numerical))
    print(z)
    df_numerical = df_numerical[(z < 3).all(axis=1)]
    print(df_numerical.count())


def scatterContinuous():
    print("\nGrafikai su stipria tiesine priklausomybe:  variklio turis/CO2 ismetimas, degalu sunaudojimas mieste/CO2 ismetimas")
    ax1 = df_numerical.plot.scatter(x='Engine_Size', y='CO2_Emissions')
    ax2 = df_numerical.plot.scatter(x='Fuel_Consumption_City', y='CO2_Emissions')
    print("\nGrafikai su silpna tiesine priklausomybe: variklio turis/degalu sunaudojimas greitkelyje, cilindrai/degalu sunaudojimas greitkelyje")
    ax3 = df_numerical.plot.scatter(x='Fuel_Consumption_Highway', y='Engine_Size')
    ax4 = df_numerical.plot.scatter(x='Fuel_Consumption_Highway', y='Cylinders')
    plt.show()


def fuelType_TransmissionGraph():
    c2 = df_categorical['Vehicle_Class'].value_counts().plot(kind='bar')
    c1 = df_categorical.query('`Fuel_Type` == "Regular"')
    c1 = c1.groupby(['Transmission']).size().reset_index(name='counts')
    ax5 = c1.plot.bar(x="Transmission", y="counts", rot=0)
    plt.title("'Transmission' when 'Fuel Type' == Regular")

    c3 = df_categorical.query('`Fuel_Type` == "Diesel"')
    c3 = c3.groupby(['Transmission']).size().reset_index(name='counts')
    ax6 = c3.plot.bar(x="Transmission", y="counts", rot=0)
    plt.title("'Transmission' when 'Fuel Type' == Diesel")


def fuelType_VehicleClassGraph():
    c8 = df_categorical['Fuel_Type'].value_counts().plot(kind='bar')
    c5 = df_categorical.query('`Fuel_Type` == "Premium"')
    c5 = c5.groupby(['Vehicle_Class']).size().reset_index(name='counts')
    ax10 = c5.plot.bar(x="Vehicle_Class", y="counts", rot=0)
    plt.title("'Vehicle Class' when 'Fuel Type' == Premium")
    c6 = df_categorical.query('`Fuel_Type` == "Regular"')
    c6 = c6.groupby(['Vehicle_Class']).size().reset_index(name='counts')
    ax11 = c6.plot.bar(x="Vehicle_Class", y="counts", rot=0)
    plt.title("'Vehicle Class' when 'Fuel Type' == Regular")


# Rysys tarp tolydiniu atributu
def cov_and_corr():
    global corr
    print("\nKovariacija:")
    print(df.cov())
    corr = df.corr()
    print("\nKoreliacija:")
    print(corr)
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr, annot=True)
    plt.show()


if __name__ == '__main__':
    
    plt.close('all')
    print(" ")
    # Nuskaito pradinius duomenis
    df, categorical, numerical = createDataset()

    print(categorical)
    print(numerical)
    print(" ")

    # 2 Skaitinio tipo
    continuousAnalysis()

    # 2 Kategorinio tipo
    categoricalAnalysis()

    # 4
    df.hist()
    plt.show()
    # 5-6
    Outliers()

    # 7.1 Tolydinio tipo scatter plot
    scatterContinuous()

    # 7.2 Scatter Plot Matrix
    pd.plotting.scatter_matrix(df_numerical, alpha=0.2)
    plt.show()

    # 7.3 Kategorinio tipo bar plot

    # Fuel Type - Transmission
    fuelType_TransmissionGraph()

    # Fuel Type - Vehicle Class

    fuelType_VehicleClassGraph()

    # 7.4 Kategoriniai ir tolydiniai

    # Histogramos
    df.hist(column=['CO2_Emissions'], by='Fuel_Type')
    df.hist(column=['CO2_Emissions'], by='Vehicle_Class')

    # Boxplot
    # zalia rodykle per viduri - mediana
    # staciakampio krastines - pirma(apacioje) ir trecia kvartilius(virsuje)
    # horiz. tieses - min ir max
    # uz ribu - outliers

    df.boxplot(column=['CO2_Emissions'], by='Fuel_Type')
    df.boxplot(column=['CO2_Emissions'], by='Vehicle_Class')

    # 8 Kovariacija ir koreliacija
    cov_and_corr()

