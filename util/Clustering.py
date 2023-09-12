#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import time
import os.path
# from sklearnex import patch_sklearn, unpatch_sklearn 
# patch_sklearn()
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift


# values: a two-dimensional array, m number of n-dimensional vectors to be clustered;
def modified_kmeans_cluster(values_to_cluster, threshold, k_start, n_clusters=None):
    if n_clusters is not None:
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(values_to_cluster)
        return  kmeans.labels_
    else:
        n_clusters = k_start
        n_values = len(values_to_cluster)
        assert n_values > 0
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(values_to_cluster)
        inertias = [kmeans.inertia_]
        while n_values > n_clusters:
            n_clusters_new = n_clusters + 1
            kmeans_new = KMeans(n_clusters=n_clusters_new, n_init="auto", random_state=0).fit(values_to_cluster)
            inertias.append(kmeans_new.inertia_)
            if terminate_clustering(inertias, threshold):
                break
            kmeans = kmeans_new
            n_clusters += 1
        return kmeans.labels_


def terminate_clustering(inertias, threshold):
    # method: compute relative improvement toward previous step
    assert len(inertias) > 1
    improvement = 1 - (inertias[-1] / inertias[-2])
    return improvement < threshold




def cluster_existed_features(network_folder_path, classes, layers_indexes, taus):
    appendixes = ["_correctly_classified_features.csv", "_incorrectly_classified_features.csv"]
    product = ((i, y, appendix) for i in layers_indexes for y in classes for appendix in appendixes)
    
    for i, y, appendix in product:
        start_time = time.time()
        # load data for class y at layer minus i
        features_file_path = network_folder_path +"Layer_minus_" + str(i) + "/class_" + str(y) + appendix
        df = pd.read_csv(features_file_path)
        index_values = df["index"].to_numpy()
        values_to_cluster = df[df.columns[3:]].to_numpy()
        
        if len(values_to_cluster):
            # specify path and then load existing clustering results
            k_and_taus = dict()
            taus_existed = []
            clustering_results = pd.DataFrame(df, columns=["index", "true_label", "pred_label"])
            clustering_results_path = network_folder_path + "Layer_minus_" + str(i) + "/clustering_results_class_" + str(y) + appendix

            if os.path.exists(clustering_results_path):
                clustering_results = pd.read_csv(clustering_results_path)
                for col in clustering_results.columns[3:]:
                    k_and_taus[col] = clustering_results[col].max() + 1

            # update the existing values of tau
            taus_existed = [float(key) for key in k_and_taus.keys()]

            # remove existing tau from list existed_taus
            taus_new = [tau for tau in taus if tau not in taus_existed]

            # iterate every tau to cluster the given data
            for tau in taus_new:
                # fix starting searching point
                k_start = 1
                bigger_taus = [x for x in taus_existed if x > tau]
                if len(bigger_taus):
                    tau_closest = min(bigger_taus) 
                    k_start = k_and_taus[str(tau_closest)]

                # start to cluster
                cluster_labels = modified_kmeans_cluster(values_to_cluster, tau, k_start)
                clustering_results[str(tau)] = cluster_labels
                taus_existed.append(tau)
                k_and_taus[str(tau)] = max(cluster_labels) + 1

            clustering_results.to_csv(clustering_results_path, index = False)
            elapsed_time = time.time() - start_time
            print("file:" + "Layer_minus_" + str(i) + "_class_" + str(y) + appendix + ",", "lasting time:", elapsed_time, "seconds")


def features_clustering(features, taus):
    start_time = time.time()
    values_to_cluster = features
        
    if len(values_to_cluster):
        # specify path and then load existing clustering results
        k_and_taus = dict()
        taus_existed = []
        

        # if os.path.exists(clustering_results_path):
        #     clustering_results = pd.read_csv(clustering_results_path)
        #     for col in clustering_results.columns[3:]:
        #         k_and_taus[col] = clustering_results[col].max() + 1
        # else:
        #     clustering_results = pd.DataFrame()

        # update the existing values of tau
        taus_existed = [float(key) for key in k_and_taus.keys()]

        # remove existing tau from list existed_taus
        taus_new = [tau for tau in taus if tau not in taus_existed]
        clustering_results = dict()
        # iterate every tau to cluster the given data
        for tau in taus_new:
            # fix starting searching point
            k_start = 1
            bigger_taus = [x for x in taus_existed if x > tau]
            if len(bigger_taus):
                tau_closest = min(bigger_taus) 
                k_start = k_and_taus[str(tau_closest)]

            # start to cluster
            cluster_labels = modified_kmeans_cluster(values_to_cluster, tau, k_start)
            clustering_results[str(tau)] = cluster_labels
            taus_existed.append(tau)
            k_and_taus[str(tau)] = max(cluster_labels) + 1

        # clustering_results.to_csv(clustering_results_path, index = False)
        elapsed_time = time.time() - start_time
        # print("clustering time:", elapsed_time, "seconds")
        return clustering_results


