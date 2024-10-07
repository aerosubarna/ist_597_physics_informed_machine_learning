import numpy as np
import random
import matplotlib.pyplot as plt

def initialize_centroids_randomly(data, num_clusters):

    # randomly initialize the centroids
    # select any k points from the data as the initial centroids
    random_indices = random.sample(range(len(data)), num_clusters)
    centroids = data[random_indices]

    return centroids

def initialize_centroids_plusplus(data, num_clusters):
    
    #select the first centroid randomly
    centroids = []
    random_index = random.randint(0, len(data)-1)
    centroid = data[random_index]
    centroids.append(centroid)

    # calculate the distance between all other points and already selected centroids
    # take the minimum of the distances between each point and the centroids
    # select the next centroid based on the probability of the distance
    for i in range(num_clusters-1):
        min_distances_list = []
        probabilities = [] 
        for data_point in data:
            distances = []
            for centroid in centroids:
                distance = np.linalg.norm(centroid - data_point)
                distances.append(distance)
            min_distance = np.min(distances)
            min_distances_list.append(min_distance)
        probabilities = min_distances_list / np.sum(min_distances_list)
        new_centroid_index = np.random.choice(len(data), 1, p=probabilities)
        new_centroid = data[new_centroid_index]
        centroids.append(new_centroid)
    
    return centroids
    
def update_centroids(data, centroids):

    # calculate the distance between each point and all the centroids
    # assign each data point to the closest centroid
    cluster_index_list = np.zeros(len(data))
    for i in range(len(data)):
        distances = []
        for centroid in centroids:
            distance = np.linalg.norm(centroid - data[i])
            distances.append(distance)
        closest_centroid_index = np.argmin(distances)
        cluster_index_list[i] = closest_centroid_index

    # update the centroids
    # calculate the mean of the data points in each cluster
    # set the mean as the new centroid
    clusters = np.empty(len(centroids), dtype=object)
    for i in range(num_clusters):
        cluster_data = data[np.where(cluster_index_list == i)]
        centroids[i] = np.mean(cluster_data, axis=0)
        clusters[i] = cluster_data

    return centroids, clusters

def k_means_clustering(data, num_clusters):
    
    while True:
        # repeat until the centroids do not change
        prev_centroids = initialize_centroids_randomly(data, num_clusters)
        centroids, clusters = update_centroids(data, prev_centroids)
        if np.array_equal(prev_centroids, centroids):
            break

    return centroids, clusters

def k_means_plusplus_clustering(data, num_clusters):
    
    while True:
        # repeat until the centroids do not change
        prev_centroids = initialize_centroids_plusplus(data, num_clusters)
        centroids, clusters = update_centroids(data, prev_centroids)
        if np.array_equal(prev_centroids, centroids):
            break

    return centroids, clusters

# test

num_clusters = 2
# generate gaussian blobs which are sufficiently separated
data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
data2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100)
data = np.concatenate((data1, data2), axis=0)

# centroids, clusters = k_means_clustering(data, num_clusters)
# print(centroids)
# print(clusters)

centroids, clusters = k_means_plusplus_clustering(data, num_clusters)


# plot the data points and the centroids
plt.scatter(data[:, 0], data[:, 1], c='black', s=7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)
plt.show()




