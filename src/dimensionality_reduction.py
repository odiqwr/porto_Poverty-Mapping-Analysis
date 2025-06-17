# Handling Outliers with Uniform Manifold Approximation and Projection Dimensionality Reduction

# 1. Load CSV dan drop column 'provinsi'
X = df.drop(columns=['Provinsi'])

# 2. Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Graph K-Nearest Neighbors
n_neighbors = 15
knn = NearestNeighbors(n_neighbors=n_neighbors)
knn.fit(X_scaled)
distances, indices = knn.kneighbors(X_scaled)  # (n_samples, n_neighbors)

# 4. Calculate p and q parameter
rhos = distances[:, 1]  # ρ_i = jarak ke tetangga terdekat (selain diri sendiri)
sigmas = []
log_k = np.log2(n_neighbors)
for i in range(len(X_scaled)):
    def loss_fn(sigma):
        if sigma <= 0:
            return 1e10
        psum = np.sum(np.exp(-(distances[i, 1:] - rhos[i]) / sigma))
        return (psum - log_k)**2
    result = scipy.optimize.minimize_scalar(loss_fn, bounds=(1e-3, 10.0), method='bounded')
    sigmas.append(result.x)
sigmas = np.array(sigmas)

# 5. Calculate asimetris weight between points
n_samples = X_scaled.shape[0]
P = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(1, n_neighbors):  # skip j=0 (diri sendiri)
        neighbor_idx = indices[i, j]
        distance_ij = distances[i, j]
        weight = np.exp(-(distance_ij - rhos[i]) / sigmas[i])
        P[i, neighbor_idx] = weight

# 6. Combine weights into fuzzy simple sets
P_symmetric = P + P.T - P * P.T

# 7. The first combining
np.random.seed(42)
Y = np.random.normal(0, 1, size=(n_samples, 2))  # 2D embedding awal

# 8. Convert P and Y to tensor
P_tensor = torch.tensor(P_symmetric, dtype=torch.float32)  # from steps 6
Y_torch = torch.tensor(Y, dtype=torch.float32, requires_grad=True)  # from steps 7

# 9. UMAP Parameters
a = 1.929
b = 0.7915

# 10. Define loss function
def umap_loss(Y, P, a=1.929, b=0.7915):
    eps = 1e-4  # untuk menghindari log(0)
    n = Y.shape[0]
    # Hitung jarak euclidean antar semua titik
    dist_squared = torch.cdist(Y, Y, p=2) ** 2  # (n x n)
    # Hitung q_ij
    Q = 1.0 / (1.0 + a * dist_squared ** b)
    Q = torch.clamp(Q, min=eps, max=1.0 - eps)  # Hindari log(0)
    # UMAP cross-entropy loss
    CE = P * torch.log(Q) + (1 - P) * torch.log(1 - Q)
    return -torch.sum(CE)  # Negatif karena kita mau meminimalkan

# 11. Optimization with gradient descent
optimizer = optim.Adam([Y_torch], lr=0.1)
for epoch in range(500):
    optimizer.zero_grad()
    loss = umap_loss(Y_torch, P_tensor, a, b)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
      
# 12. Take final combination
Y_final = Y_torch.detach().numpy()

# 13. Display visualization for final combination
plt.figure(figsize=(10, 6))
plt.scatter(Y_final[:, 0], Y_final[:, 1], c='blue', s=10, alpha=0.7)
plt.title("Manual UMAP Embedding (Optimized)")
plt.xlabel("Dimensi 1")
plt.ylabel("Dimensi 2")
plt.grid(True)
plt.show()

# 14. Checking Outliers After Dimensionality Reduction
featuresNum = ['Persentase Pelajar_Bekerja','Persentase Pelajar_Belajar','Persentase Tidak/Belum Sekolah','Pekerja Formal (%)','Pekerja Informal (%)','Jumlah Sekolah Atas dan Kejuruan','Jumlah Kampus (Unit)','Jumlah Industri Kecil n Mikro (Unit)','Jumlah Industri Besar n Sedang (Unit)','Upah Minimum Provinsi']
plt.figure(figsize=(15, 7))
for i in range(0, len(featuresNum)):
    plt.subplot(1, len(featuresNum), i+1)
    sns.boxplot(y=df[featuresNum[i]], color='green', orient='v')
    plt.tight_layout()
