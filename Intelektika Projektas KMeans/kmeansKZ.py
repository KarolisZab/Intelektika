from matplotlib import colors
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

df = pd.read_csv('credit_risk_dataset.csv').fillna(0)[:500]
pd.plotting.scatter_matrix(df)
plt.show()


# Atspausdinti trukstamu reiksmiu kieki kiekviename stulpelyje
df.isnull().sum()

# Pasalinti trukstamas reiksmes
df.dropna(axis=0, inplace=True)

column = "loan_amnt"
b = "loan_status"
#rows = ["person_age", "person_income", "loan_int_rate"]
rows = ["person_income", "loan_int_rate", "loan_status"]


def centroids(K, b):
    X = df[[column, b]]
    Centroids = (X.sample(n=K))

    total_inertia = 0

    diff = 1
    j = 0

    while(diff != 0):
        Xd = X
        i = 1

        for index1, row_c in Centroids.iterrows():
            Ed = []

            for index2, row_d in Xd.iterrows():
                d1 = (row_c[column]-row_d[column])**2
                d2 = (row_c[b]-row_d[b])**2
                d = np.sqrt(d1+d2)
                Ed.append(d)
            X = X.copy()
            X.loc[:, i] = Ed
            i = i+1

        C = []

        for index, row in X.iterrows():
            min_dist = row[1]
            pos = 1
            for i in range(K):
                if row[i+1] < min_dist:
                    min_dist = row[i+1]
                    pos = i+1
            C.append(pos)
        X = X.copy()
        X.loc[:, "Cluster"] = C
        Centroids_new = X.groupby(["Cluster"]).mean()[
            [b, column]]
        if j == 0:
            diff = 1
            j = j+1
        else:
            diff = (Centroids_new[b] - Centroids[b]).sum() + \
                (Centroids_new[column] - 
                 Centroids[column]).sum()
        Centroids = X.groupby(["Cluster"]).mean()[
            [b, column]]

    color = ['blue', 'green', 'cyan', 'purple', 'orange']

    for k in range(K):
        data = X[X["Cluster"] == k + 1]
        other_clusters = X[X["Cluster"] != k + 1]

        silhouette_coeffs = []

        # centroid, will shift to the new position, which is the actual centroid for each of these groups

        for i, sample in data.iterrows():
            ai = calculate_ai(sample, data, column, b)
            bi = calculate_bi(sample, other_clusters, column, b)
            # set of all points assigned to the ith cluster
            Si = (bi - ai) / max(bi, ai)
            silhouette_coeffs.append(Si)

        silhouette[b].insert(k, silhouette_coeffs)

        centroid = Centroids.iloc[k]
        total_inertia += calculate_inertia(centroid, data, column, b)

    inertia[b].append(total_inertia)

    #print(f"Inertia for {b}: {inertia[b]}")

    return X

# calculates the average distance between a given sample point and all other points within the same cluster
# computes the average distance from a given sample point to all other points within the same cluster, 
# which is used in calculating the silhouette coefficient for the sample point.
def calculate_ai(sample, data, column, row):
    x1 = sample[column]
    y1 = sample[row]

    ai = 0

    for i, row_data in data.iterrows():
        x2 = row_data[column]
        y2 = row_data[row]
        ai += euclidean(x1, y1, x2, y2)

    # avg distance
    return ai / len(data)

# calculates the average distance between a given sample point and the points in the nearest neighboring clusters.
# calculates the average distance from a given sample point to the points in the nearest neighboring clusters 
# and returns the minimum of these average distances
def calculate_bi(sample, other_clusters, column, row):
    x1 = sample[column]
    y1 = sample[row]

    clusters = other_clusters.Cluster.unique()

    mean_distances = []

    for cluster in clusters:
        data = other_clusters[other_clusters.Cluster == cluster]
        mean_distance = 0
        for i, row_data in data.iterrows():
            x2 = row_data[column]
            y2 = row_data[row]
            mean_distance += euclidean(x1, y1, x2, y2)
        mean_distances.append(mean_distance / len(data))

    return min(mean_distances)


def calculate_inertia(centroid, data, column, row):
    total_inertia = 0

    x2 = centroid[column]
    y2 = centroid[row]
    for i, row_data in data.iterrows():
        x1 = row_data[column]
        y1 = row_data[row]
        total_inertia += euclidean(x1, y1, x2, y2)

    return total_inertia


def euclidean(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def plot_silhouette_graphs(K, silhouette_values, data, column, row):
    X = 0
    for i in silhouette_values:
        X += len(i)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, X + (K + 1) * 10])

    y_lower = 10

    for i in range(K):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_values[i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = len(ith_cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / K)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silueto koeficientų atvaizdavimas su visais klasteriais")
    ax1.set_xlabel("Silueto koeficientų reikšmės")
    ax1.set_ylabel("klasteris")

    # The vertical line for average silhouette score of all the values
    avg = sum(ith_cluster_silhouette_values) / len(ith_cluster_silhouette_values)
    print(column, row, K, avg)
    ax1.axvline(x=avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    Xlen = 0
    for i in range(K):
        color = cm.nipy_spectral(float(i) / K)
        current_data = data[data["Cluster"] == i+1]
        Xlen += len(current_data)
        ax2.scatter(current_data[column].to_numpy(), current_data[row].to_numpy(), marker='.', s=30, lw=0, alpha=0.7,
                    facecolor=color, edgecolor='k')

        centerX = np.average(current_data[column])
        centerY = np.average(current_data[row])
        ax2.scatter(centerX, centerY, marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        ax2.scatter(centerX, centerY, marker='$%d$' % (i+1), alpha=1,
                    s=50, edgecolor='k')

    ax1.set_ylim([0, Xlen + (K + 1) * 10])
    # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers

    ax2.set_title("Klasterizavimo taškų vizualizacija")
    ax2.set_xlabel(f"Reikšmė: {column}")
    ax2.set_ylabel(f"Reikšmė: {row}")

    plt.suptitle(("K-vidurkių klasterizavimo silueto analizė "
                  "kai K = %d" % K),
                 fontsize=14, fontweight='bold')

    plt.show()


# def plot_inertia(inertia, row):
#     #xs = np.arange(start, end, step)
#     k = 3
#     plt.plot(k, inertia)
#     plt.title(f'Reikšmės {row} inercijos')
#     plt.ylabel('Inercija')
#     plt.xlabel('Klasterių skaičius')
#     plt.show()

inertia = {
    "person_income": [],
    "loan_int_rate": [],
    "loan_status": [],
}

silhouette = {
    "person_income": [],
    "loan_int_rate": [],
    "loan_status": [],
}

for K in range(3, 4):
    for row in rows:
        X = centroids(K, row)
        plot_silhouette_graphs(K, silhouette[row], X, 'loan_amnt', row)


### test inertia
# for K in range(3, 6):
#     for row in rows:
#         #row = 'person_income'
#         X = centroids(K, row)
#         plot_inertia(inertia[row], row)

# for row in rows:
#     plot_inertia(inertia[row], row)

