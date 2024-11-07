import cv2
import numpy as np
import matplotlib.pyplot as plt

# Bước 1: Đọc ảnh và chuyển sang không gian màu RGB
img_path = '/bai6/leon.jpg'  # Thay thế bằng đường dẫn ảnh của bạn
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Bước 2: Thu nhỏ ảnh để tăng tốc độ xử lý
image_small = cv2.resize(image_rgb, (image_rgb.shape[1] // 8, image_rgb.shape[0] // 8))
pixels_small = image_small.reshape(-1, 3)

# Hàm tính khoảng cách Euclidean
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Hàm K-means đơn giản
def kmeans_custom(pixels, K, max_iters=100):
    # Khởi tạo ngẫu nhiên các tâm cụm
    np.random.seed(42)
    centers = pixels[np.random.choice(pixels.shape[0], K, replace=False)]

    for _ in range(max_iters):
        # Bước 1: Gán mỗi điểm dữ liệu vào cụm gần nhất
        clusters = [[] for _ in range(K)]
        for pixel in pixels:
            distances = [euclidean_distance(pixel, center) for center in centers]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(pixel)

        # Bước 2: Tính lại tâm cụm
        new_centers = []
        for cluster in clusters:
            if cluster:  # Nếu cụm không rỗng
                new_center = np.mean(cluster, axis=0)
                new_centers.append(new_center)
            else:  # Nếu cụm rỗng, giữ nguyên tâm cụm cũ
                new_centers.append(centers[len(new_centers)])
        
        new_centers = np.array(new_centers)

        # Kiểm tra hội tụ
        if np.all(centers == new_centers):
            break
        centers = new_centers

    # Tạo nhãn cho mỗi pixel
    labels = np.zeros(len(pixels), dtype=int)
    for i, pixel in enumerate(pixels):
        distances = [euclidean_distance(pixel, center) for center in centers]
        labels[i] = np.argmin(distances)

    return centers, labels

# Khởi tạo các giá trị K cho K-means
K_values = [2, 3, 4, 5]

# Bước 3: Áp dụng thuật toán K-means tự xây dựng và hiển thị kết quả
fig, axs = plt.subplots(1, len(K_values), figsize=(20, 5))
fig.suptitle("K-means Clustering with Different K Values (Custom Implementation)")

for idx, K in enumerate(K_values):
    centers, labels = kmeans_custom(pixels_small, K)
    
    # Thay thế các giá trị pixel bằng tâm cụm tương ứng
    clustered_pixels = np.array([centers[label] for label in labels])
    clustered_image = clustered_pixels.reshape(image_small.shape).astype(np.uint8)
    
    # Hiển thị ảnh phân cụm
    axs[idx].imshow(clustered_image)
    axs[idx].axis('off')
    axs[idx].set_title(f'K = {K}')

plt.show()
