import os
import random
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

# Đường dẫn đến thư mục chứa ảnh nha khoa
dental_images_path = './Periapical Lesions/Augmentation JPG Images'

# Lấy danh sách tất cả các ảnh trong thư mục
all_images = [f for f in os.listdir(dental_images_path) if f.endswith('.jpg')]
print(f'Total dental images found: {len(all_images)}')  # Kiểm tra số lượng ảnh
# Chọn ngẫu nhiên 300 ảnh
selected_images = random.sample(all_images, 300)

# Tạo dữ liệu nha khoa với ảnh đã chọn
dental_data = []
for img_name in selected_images:
    img_path = os.path.join(dental_images_path, img_name)
    img = imread(img_path)
    img_resized = resize(rgb2gray(img), (64, 64)).flatten()  # Chuyển đổi kích thước ảnh và chuyển sang grayscale
    label = 0 if 'class_0' in img_name else 1  # Gán nhãn dựa trên tên file (giả định)
    dental_data.append(list(img_resized) + [label])  # Thêm nhãn vào dữ liệu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data_dental = dental_data[:210]  # 70% cho huấn luyện
test_data_dental = dental_data[210:]    # 30% cho kiểm tra

# In ra thông tin dữ liệu
print(f'Train data size: {len(train_data_dental)}')
print(f'Test data size: {len(test_data_dental)}')

# Hàm tính Gini Index
def gini_index(groups, classes):
    total_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion * proportion
        gini += (1.0 - score) * (size / total_instances)
    return gini

# Tạo cây CART (ví dụ đơn giản)
def cart(train):
    classes = list(set(row[-1] for row in train))  # Lớp duy nhất
    # Thêm logic để chia nhóm và phát triển cây ở đây
    return "Cây CART (Gini Index) đã được tạo"

# Huấn luyện mô hình CART
cart_model = cart(train_data_dental)
print(cart_model)

# Kiểm tra độ chính xác trên tập kiểm tra
def predict_cart(test):
    # Thêm logic để dự đoán ở đây
    return [0] * len(test)  # Ví dụ dự đoán giả (tất cả về lớp 0)

predictions_cart = predict_cart(test_data_dental)
accuracy_cart = sum(1 for i in range(len(predictions_cart)) if predictions_cart[i] == test_data_dental[i][-1]) / len(test_data_dental)
print(f'CART Accuracy on Dental Data: {accuracy_cart}')

# Tạo cây ID3 (Information Gain)
def id3(train):
    classes = list(set(row[-1] for row in train))  # Lớp duy nhất
    # Thêm logic để chia nhóm và phát triển cây ở đây
    return "Cây ID3 (Information Gain) đã được tạo"

# Huấn luyện mô hình ID3
id3_model = id3(train_data_dental)
print(id3_model)

# Kiểm tra độ chính xác trên tập kiểm tra
def predict_id3(test):
    # Thêm logic để dự đoán ở đây
    return [0] * len(test)  # Ví dụ dự đoán giả (tất cả về lớp 0)

predictions_id3 = predict_id3(test_data_dental)
accuracy_id3 = sum(1 for i in range(len(predictions_id3)) if predictions_id3[i] == test_data_dental[i][-1]) / len(test_data_dental)
print(f'ID3 Accuracy on Dental Data: {accuracy_id3}')