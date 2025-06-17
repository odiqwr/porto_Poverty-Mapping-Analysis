# Function of Density Based Spatial Clustering of Application with Noise

# --- Euclidean distance function ---
def euclidean_distance(a, b):
    # Compute the Euclidean distance between two points
    return np.linalg.norm(a - b)
  
# --- Find all neighbors within eps radius ---
def region_query(data, point_idx, eps):
    neighbors = []
    for i, point in enumerate(data):
        # If the distance between the current point and another point is within eps
        if euclidean_distance(data[point_idx], point) <= eps:
            neighbors.append(i)  # Add index of neighboring point
    return neighbors
  
# --- Expand a new cluster starting from a core point ---
def expand_cluster(data, labels, point_idx, cluster_id, eps, minPts):
    # Get all neighboring points within eps radius
    neighbors = region_query(data, point_idx, eps) 
    # Check if the number of neighbors is less than MinPts
    if len(neighbors) < minPts:
        labels[point_idx] = -1  # Mark as noise
        return False
    else:
        # Assign the cluster ID to the seed point
        labels[point_idx] = cluster_id
        # Initialize queue with neighbors for BFS expansion
        queue = deque(neighbors)
        while queue:
            curr_idx = queue.popleft()
            # If the point was previously marked as noise, now include it in the cluster
            if labels[curr_idx] == -1:
                labels[curr_idx] = cluster_id
            # If the point is already assigned to a cluster, skip it
            if labels[curr_idx] is not None:
                continue
            # Assign current point to the cluster
            labels[curr_idx] = cluster_id
            # Check neighbors of the current point
            curr_neighbors = region_query(data, curr_idx, eps)
            # If the current point is also a core point, expand further
            if len(curr_neighbors) >= minPts:
                queue.extend(curr_neighbors)  # Add new neighbors to the queue
        return True
      
# --- Main DBSCAN clustering function ---
def dbscan(data, eps, minPts):
    # Initialize all labels as None (unvisited)
    labels = [None] * len(data)
    # Cluster ID starts from 0
    cluster_id = 0
    # Iterate over all points in the dataset
    for point_idx in range(len(data)):
        # If the point is already visited/labeled, skip it
        if labels[point_idx] is not None:
            continue
        # Attempt to grow a new cluster from this point
        if expand_cluster(data, labels, point_idx, cluster_id, eps, minPts):
            cluster_id += 1  # Increment cluster ID for the next cluster
    # Return the resulting cluster labels as a NumPy array
    return np.array(labels)
----------------------------------------------------------------------------Implementation-----------------------------------------------------------------------------
# 1. Load umap dataset
data = pd.read_csv("umap_result.csv")
X = data[['UMAP1', 'UMAP2']].values

# 2. Parameter grid
eps_values = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
minpts_values = [5, 10, 15, 20]

# 3. display result of clustering
results = []
iteration = 1

# 4. iteration of clustering dbscan function with eps, minpts, clusters, nouse, silhouette, and dbi results
for eps in eps_values:
    for minpts in minpts_values:
        labels = dbscan(X, eps, minpts)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        # check valid number of clusters
        if n_clusters > 1:
            ss = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
        else:
            ss = -1
            dbi = float('inf')
        # display results on dbscan clustering function iterations
        print(f"Iterasi {iteration}: eps={eps}, minPts={minpts}")
        print(f"  ➤ Cluster: {n_clusters}, Noise: {n_noise}")
        print(f"  ➤ Silhouette Score: {ss:.4f}")
        print(f"  ➤ Davies-Bouldin Index: {dbi:.4f}\n")
        # update values ​​on cluster iteration results
        results.append({
            'iteration': iteration,
            'eps': eps,
            'minpts': minpts,
            'clusters': n_clusters,
            'noise': n_noise,
            'silhouette': ss,
            'dbi': dbi,
            'labels': labels
        })
        # for looping
        iteration += 1

# Determining best parameters dbscan with 
valid_results = [r for r in results if r['clusters'] > 1]

# Strategy for selecting the best clustering results:
# - Silhouette tinggi
# - DBI rendah
# - noise rendah

if valid_results:
    # Skor kombinasi (bisa dimodifikasi sesuai bobot preferensi)
    for r in valid_results:
        r['score'] = r['silhouette'] - 0.1 * r['dbi']
    # determining best cluster results
    best_result = max(valid_results, key=lambda r: r['score'])
    # display best cluster results
    print("=== Hasil Terbaik ===")
    print(f"Iterasi: {best_result['iteration']}")
    print(f"  ➤ eps={best_result['eps']}, minPts={best_result['minpts']}")
    print(f"  ➤ Cluster: {best_result['clusters']}, Noise: {best_result['noise']}")
    print(f"  ➤ Silhouette: {best_result['silhouette']:.4f}")
    print(f"  ➤ DBI: {best_result['dbi']:.4f}")
else:
    print("Tidak ditemukan hasil klasterisasi yang valid.")
