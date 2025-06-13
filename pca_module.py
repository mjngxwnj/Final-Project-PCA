# 1. Subtract the mean of each variable
# 2. Calculate the Covariance Matrix
# 3. Compute the Eigenvalues and Eigenvectors
# 4. Sort Eigenvalues in descending order
# 5. Select a subset from the rearranged Eigenvalue matrix
# 6. Transform the data

import numpy as np
import pandas as pd 

class PCA:
    def __init__(self, n_components=0.8):
        self.n_components = n_components

    def compute_mean(self, X):
        """
        mean = 1/n * sum(xi) with i: {1,...,n}
        Trả về vector trung bình của từng cột (feature-wise mean)
        """
        n_samples = len(X)
        n_features = len(X[0])
        mean = []
        for j in range(n_features):
            feature_sum = 0
            for i in range(n_samples):
                feature_sum += X[i][j]
            mean.append(feature_sum / n_samples)
        return mean

    def center_data(self, X, mean_vector):
        """
        Trừ vector trung bình khỏi mỗi sample trong X.
        X: ma trận (n_samples x n_features)
        mean_vector: vector trung bình (length = n_features)
        """
        n_samples = len(X)
        n_features = len(X[0])
        centered_data = []
        for i in range(n_samples):
            row = []
            for j in range(n_features):  
                row.append(X[i][j] - mean_vector[j])
            centered_data.append(row)
        return centered_data

    def compute_covariance_matrix(self, X_centered):
        """
        Tạo ma trận hiệp phương sai theo công thức gốc với đầu vào X_centered được tính trước (X - mean)
        Đầu ra là ma trận hiệp phương sai 
        """
        n_samples = len(X_centered)
        n_features = len(X_centered[0])
        cov_matrix = []

        for i in range(n_features):
            row = []
            for j in range(n_features):
                cov_ij = 0
                for k in range(n_samples):
                    cov_ij += X_centered[k][i] * X_centered[k][j]
                cov_ij /= (n_samples - 1) 
                row.append(cov_ij)
            cov_matrix.append(row)
        return cov_matrix

    def compute_eigen_decomposition(self, cov_matrix):
        """
        Tính trị riêng và vector riêng từ ma trận hiệp phương sai.
        Chuẩn hóa dấu của các vector riêng để nhất quán hướng.
        Trả về danh sách các cặp (trị riêng, vector riêng) đã được sắp xếp giảm dần theo trị tuyệt đối.
        """
        # B1: Dùng numpy để tính trị riêng và vector riêng
        eig_vals, eig_vecs = np.linalg.eig(np.array(cov_matrix))  # eig_vecs: cột là vector riêng

        # B2: Chuyển các vector riêng thành từng hàng cho dễ xử lý (mỗi hàng là 1 vector riêng)
        eig_vecs = eig_vecs.T

        # B3: Tìm chỉ số phần tử có giá trị tuyệt đối lớn nhất trong từng vector riêng
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=1)

        # B4: Tạo danh sách dấu tại phần tử lớn nhất để chuẩn hóa hướng vector riêng
        signs = []
        for i in range(len(eig_vecs)):
            sign = np.sign(eig_vecs[i][max_abs_idx[i]])
            signs.append(sign)

        # B5: Chuẩn hóa hướng vector riêng bằng cách nhân với dấu tương ứng
        for i in range(len(eig_vecs)):
            eig_vecs[i] = eig_vecs[i] * signs[i]

        # B6: Tạo danh sách các cặp (trị riêng, vector riêng)
        eig_pairs = []
        for i in range(len(eig_vals)):
            pair = [eig_vals[i], eig_vecs[i]]  # tạo cặp trị riêng và vector riêng tương ứng
            eig_pairs.append(pair)

        # B7: Sắp xếp các cặp theo trị tuyệt đối của trị riêng (giảm dần)
        for i in range(len(eig_pairs)):
            for j in range(i + 1, len(eig_pairs)):
                if abs(eig_pairs[j][0]) > abs(eig_pairs[i][0]):
                    eig_pairs[i], eig_pairs[j] = eig_pairs[j], eig_pairs[i]

        return eig_pairs

    def select_top_components(self, eigenpairs, k):
        eig_vals_sorted = []
        eig_vecs_sorted = []

        for i in range(k): 
            eig_vals_sorted.append(eigenpairs[i][0])
            eig_vecs_sorted.append(eigenpairs[i][1])

        return eig_vals_sorted, eig_vecs_sorted

    def fit(self, X):
        self.mean_vector = self.compute_mean(X)
        X_centered = self.center_data(X, self.mean_vector)
        self.cov_matrix = self.compute_covariance_matrix(X_centered)
        self.eigenpairs = self.compute_eigen_decomposition(self.cov_matrix)

        eig_vals = [abs(pair[0]) for pair in self.eigenpairs]
        total_variance = sum(eig_vals)

        explained_variance_ratio = [val / total_variance for val in eig_vals]
        cum_explained_variance = []
        cumulative_sum = 0
        for r in explained_variance_ratio:
            cumulative_sum += r
            cum_explained_variance.append(cumulative_sum)

        k = 0
        while k < len(cum_explained_variance) and cum_explained_variance[k] < self.n_components:
            k += 1
        k += 1  # chọn thêm 1 component để đảm bảo >= tỉ lệ yêu cầu

        self.eig_vals, self.components = self.select_top_components(self.eigenpairs, k)

        total_selected_variance = sum(abs(val) for val in self.eig_vals)
        self.explained_variance_ratio = [abs(val) / total_selected_variance for val in self.eig_vals]
        self.cum_explained_variance = []
        cumulative_sum = 0
        for r in self.explained_variance_ratio:
            cumulative_sum += r
            self.cum_explained_variance.append(cumulative_sum)

        self.n_selected_components = k

        return self

    def transform(self, X):
        X_centered = self.center_data(X, self.mean_vector)

        # Chiếu dữ liệu vào các thành phần chính (components)
        X_proj = []
        for x in X_centered:
            # Với mỗi vector x đã center, chiếu nó lên từng thành phần chính
            proj = []
            for i in range(len(self.components)):
                dot = 0
                for j in range(len(x)):
                    dot += x[j] * self.components[i][j]  # Tính tích vô hướng giữa x và thành phần chính thứ i
                proj.append(dot)
            X_proj.append(proj)

        return X_proj

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_pca(self, Z):
        """
        - Z: Dữ liệu sau khi đã chiếu vào không gian PCA (n_samples x n_components)
        - X_approx: Dữ liệu gần đúng đã khôi phục lại (n_samples x original_features)
        """
        X_approx = []
        for z in Z:
            # B1: Chiếu ngược từ không gian PCA về không gian gốc (không có mean)
            # mỗi feature gốc được khôi phục bằng tổ hợp tuyến tính từ các thành phần chính
            row = []
            for j in range(len(self.components[0])):  
                value = 0
                for i in range(len(self.components)):
                    value += z[i] * self.components[i][j]  
                row.append(value)

            # B2: Cộng lại giá trị trung bình để đưa về tọa độ ban đầu
            row = [row[j] + self.mean_vector[j] for j in range(len(row))]

            X_approx.append(row)

        return X_approx
    # or

    def inverse_transform(self, X_proj):
        X_proj = np.array(X_proj)
        components = np.array(self.components)
        mean_vector = np.array(self.mean_vector)
        X_reconstructed = X_proj @ components + mean_vector
        return X_reconstructed

"""
How to run: 

Translate input to list 
# X = df.values.tolist()
Call PCA
# my_pca = PCA().fit(X)

X after reduce dimension using my_pca to transform 
# X_proj = my_pca.transform(X) 

# X_reconstructed = my_pca.inverse_pca(X_proj)

Gọi hàm sau khi đã có X và X_reconstructed
# mse = mean_squared_error_manual(X, X_reconstructed)

"""