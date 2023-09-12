
import numpy as np
import pandas as pd
import pickle
import os
import sys
# append the path of the parent directory
sys.path.append("..")

from abstractions import *
from runtime_monitors import *


# def monitors_offline_construction(network_name, network_folder_path, classes, layers_indexes, taus):
#     appendixes = ["_correctly_classified_features.csv", "_incorrectly_classified_features.csv"]
#     product = ((i, y) for i in layers_indexes for y in classes)

#     for i, y in product:
        
#         # load obtained features to creat reference
#         path_bad_features = network_folder_path +"Layer_minus_" + str(i) + "/class_" + str(y) + appendixes[1]
#         path_good_features = network_folder_path +"Layer_minus_" + str(i) + "/class_" + str(y) + appendixes[0]
        
#         bad_feat_clustering_results = []
#         good_feat_clustering_results = []


#         if os.path.exists(path_bad_features):
#             df_bad_features = pd.read_csv(path_bad_features)
#             bad_features_to_cluster = df_bad_features[df_bad_features.columns[3:]].to_numpy()
#             bad_features_index = df_bad_features["index"].to_numpy()
            
#             # load clustering results to partition the features
#             bad_feat_clustering_results_path = network_folder_path + "Layer_minus_" + str(i) + "/clustering_results_class_" + str(y) + appendixes[1]
#             if os.path.exists(bad_feat_clustering_results_path):
#                 bad_feat_clustering_results = pd.read_csv(bad_feat_clustering_results_path)

#         if os.path.exists(path_good_features):
#             df_good_features = pd.read_csv(path_good_features)
#             good_features_to_cluster = df_good_features[df_good_features.columns[3:]].to_numpy()
#             good_features_index = df_good_features["index"].to_numpy()
            
            
#             # load clustering results to partition the features
#             good_feat_clustering_results_path = network_folder_path + "Layer_minus_" + str(i) + "/clustering_results_class_" + str(y) + appendixes[0]
#             n_dim = good_features_to_cluster.shape[1]
#             if os.path.exists(good_feat_clustering_results_path):
#                 good_feat_clustering_results = pd.read_csv(good_feat_clustering_results_path)

#         for tau in taus:
#             good_loc_boxes = []
#             bad_loc_boxes = []

#             if len(bad_feat_clustering_results):
#                 # load clustering result related to tau
#                 bad_feat_clustering_result = bad_feat_clustering_results[str(tau)]
#                 # determine the labels of clusters
#                 bad_num_clusters = np.amax(bad_feat_clustering_result) + 1
#                 bad_clustering_labels = np.arange(bad_num_clusters)
                
#                 # extract the indices of vectors in a cluster
#                 bad_clusters_indices = []
#                 for k in bad_clustering_labels:
#                     bad_indices_cluster_k, = np.where(bad_feat_clustering_result == k)
#                     bad_clusters_indices.append(bad_indices_cluster_k)
                
#                 # creat local box for each cluster
#                 bad_loc_boxes = [Box() for i in bad_clustering_labels]
#                 for j in range(len(bad_loc_boxes)):
#                     bad_points_j = [(bad_features_index[i], bad_features_to_cluster[i]) for i in bad_clusters_indices[j]]
#                     bad_loc_boxes[j].build(n_dim, bad_points_j)
                

#             if len(good_feat_clustering_results):
#                 # load clustering result related to tau
#                 good_feat_clustering_result = good_feat_clustering_results[str(tau)]
#                 # determine the labels of clusters 
#                 good_num_clusters = np.amax(good_feat_clustering_result) + 1
#                 good_clustering_labels = np.arange(good_num_clusters)
                
#                 # extract the indices of vectors in a cluster
#                 good_clusters_indices = []
#                 for k in good_clustering_labels:
#                     good_indices_cluster_k, = np.where(good_feat_clustering_result == k)
#                     good_clusters_indices.append(good_indices_cluster_k)    
                
#                 # creat local box for each cluster
#                 good_loc_boxes = [Box() for i in good_clustering_labels]
#                 for j in range(len(good_loc_boxes)):
#                     good_points_j = [(good_features_index[i], good_features_to_cluster[i]) for i in good_clusters_indices[j]]
#                     good_loc_boxes[j].build(n_dim, good_points_j)

#             # creat the monitor for class y at layer i
#             monitor_y_i = Monitor("Box", network_name, y, i, good_ref=good_loc_boxes, bad_ref=bad_loc_boxes)
#             # save the created monitor
#             monitor_stored_folder_path = network_folder_path + "Monitors/"
#             if not os.path.exists(monitor_stored_folder_path):
#                 os.makedirs(monitor_stored_folder_path)
#             monitor_stored_path = monitor_stored_folder_path + network_name + "_monitor_for_class_" + str(y) + "_at_layer_minus_" + str(i) + "_tau_" + str(tau) + ".pkl"
#             with open(monitor_stored_path, 'wb') as f:
#                 pickle.dump(monitor_y_i, f)


def monitor_construction_from_features(features, taus, clustering_results, class_name, monitor_saving_folder):
    # if os.path.exists(clustering_result_path):
    #     clustering_results = pd.read_csv(clustering_result_path)
    # else:
    #     raise RuntimeError("Please partition your data first!")

    for tau in taus:
        loc_boxes = []

        if len(features):
            n_dim = features.shape[1]

            # load clustering result related to tau
            clustering_result = clustering_results[str(tau)]
            # determine the labels of clusters
            num_clusters = np.amax(clustering_result) + 1
            clustering_labels = np.arange(num_clusters)
            
            # extract the indices of vectors in a cluster
            clusters_indices = []
            for k in clustering_labels:
                indices_cluster_k, = np.where(clustering_result == k)
                clusters_indices.append(indices_cluster_k)
            
            # creat local box for each cluster
            loc_boxes = [Box() for i in clustering_labels]
            for j in range(len(loc_boxes)):
                points_j = [(i, features[i]) for i in clusters_indices[j]]
                loc_boxes[j].build(n_dim, points_j)
        else:
            raise RuntimeError("There exists no feature for building monitor!!")

        # creat the monitor for class y at layer i
        monitor = Monitor(good_ref=loc_boxes)
        # save the created monitor
        if not os.path.exists(monitor_saving_folder):
            os.makedirs(monitor_saving_folder)
        monitor_saving_path = monitor_saving_folder + "monitor_for_clustering_parameter" + "_tau_" + str(tau) + ".pkl"
        with open(monitor_saving_path, 'wb') as f:
            pickle.dump(monitor, f)

