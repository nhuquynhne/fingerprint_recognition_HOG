import os
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from run import run

#chuan hoa hinh anh về trung bin 0 và độc lệch chuẩn 1
def nomalise (img):
    normed = (img-np.mean(img))/(np.std(img))
    plt.imshow(normed)
    return normed

def clear():
    path = "D:\\Hoc Ky 2_2023\\CSDL ĐPT\\BTL\\feature"
    for i in os.listdir(path):
        pnpy = path + "\\" + i
        os.remove(pnpy)
        # print(pnpy)
    path = "D:\\Hoc Ky 2_2023\\CSDL ĐPT\\BTL\\ift"
    for i in os.listdir(path):
        pnpy = path + "\\" + i
        os.remove(pnpy)
        # print(pnpy)

def hog(img_gray, gx, gy, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape

    # histogram
    magnitude = np.sqrt(np.square(gx) + np.square(gy))#độ lớn gradient gx gy là căn bạc 2 tổng bình phuong
    orientation = np.arctan(np.divide(gy, gx + 0.00001))  # phương gradient(radian), cộng 0.00001 vì mẫu rất nhỏ nên để tránh chia cho 0 thì cộng them vào
    orientation = np.degrees(orientation) # chuyển radian sang độ, giá trị gốc là -90 -> 90
    orientation += 90  # cộng thêm 90 để góc chạy từ 0 đến 180 độ

    num_cell_x = w // cell_size  # sô lượng ô theo chiều ngang 248/8 =31
    num_cell_y = h // cell_size  # số lượng ô theo chiều dọc 338/8=42

    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])  # 42 x 31 x 9 khởi tạo ma trận hist_tensor với kích thước [num_cell_y, num_cell_x, bins], trong đó bins là số lượng bin trong histogram.
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy * cell_size : cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]#góc
            mag = magnitude[cy * cell_size : cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]#độ lớn

            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)  # tính histogram, kqua là 1-D vector, 9 elements = 9bins
            hist_tensor[cy, cx, :] = hist #Gán giá trị histogram (hist) cho vị trí tương ứng trong hist_tensor để lưu trữ histogram của mỗi ô
        pass
    pass

    # normalization
    redundant_cell = block_size - 1#1
    feature_tensor = np.zeros(
        [num_cell_y - redundant_cell, num_cell_x - redundant_cell, block_size * block_size * bins])#kích thước 41,30,36
    for bx in range(num_cell_x - redundant_cell):  # 30
        for by in range(num_cell_y - redundant_cell):  # 41
            by_from = by
            by_to = by + block_size
            bx_from = bx
            bx_to = bx + block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten()  # to 1-D array (vector) trải phẳng
            feature_tensor[by, bx, :] = v / linalg.norm(v, 2)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any():  # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v#chuẩn hóa

    return feature_tensor.flatten()  # 44280 features = 36x30x41

def extract(img_path, filename):
    binary_img, gx, gy = run(img_path)
    f = hog(binary_img, gx, gy)
    print('Extracted feature vector of %s. Shape:' % img_path)#đường dẫn ảnh
    print('Feature size:', f.shape)#kích thước của đặc trưng
    folderFeature = "./feature/"  + "/"#nơi lưu trữ tệp tin đặc trưng
    np.save(folderFeature + filename + "_feature.npy", f)#định dạng tệp tin đặc trưng
    print("DONE" + img_path)#thông báo DONE
    pass

def extract1(img_path, filename):
    binary_img, gx, gy = run(img_path)
    f1 = hog(binary_img, gx, gy)
    print('Extracted feature vector of %s. Shape:' % img_path)#đường dẫn ảnh
    print('Feature size:', f1.shape)#kích thước của đặc trưng
    folderFeature1 = "./ift/"  + "/"#nơi lưu trữ tệp tin đặc trưng
    np.save(folderFeature1 + filename + "_feature.npy", f1)#định dạng tệp tin đặc trưng
    print("DONE" + img_path)#thông báo DONE
    pass

def index_ouput(imgFolder, a, b, c):
    cnt = 0
    name = []
    for filename in os.listdir(imgFolder):
        cnt = cnt +1
        if (cnt == a): name.append(os.path.splitext(filename)[0])
        elif(cnt == b): name.append(os.path.splitext(filename)[0])
        elif(cnt == c): name.append(os.path.splitext(filename)[0])
    return name

def cosine_similarity(a, b):
    return  np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))#dot(a,b) là tích vô hướng 2 vector a b, norm là độ dài 2 vector a b
#độ tương đồng cosine được trả về từ hàm là một giá trị từ 0 đến 1, trong đó giá trị gần 1 cho thấy hai vector có độ tương đồng cao








if __name__ == '__main__':
    clear()
    imgFolder = "./db/PNG/"
    inp = "./db/in/"
    np.seterr(divide='ignore', invalid='ignore')

    for filename in os.listdir(imgFolder):
        extract(os.path.join(imgFolder, filename), os.path.splitext(filename)[0])
    folderFeature = "./feature/"
    features = np.array([[None] * 44280])
    for filename in os.listdir(folderFeature):
        f = np.load(folderFeature + '/' + filename)
        features = np.append(features, [f], axis=0)
    # print(features)

    for filename in os.listdir(inp):
        extract1(os.path.join(inp, filename), os.path.splitext(filename)[0])
    folderFeature1 = "./ift/"
    features1 = np.array([[None] * 44280])
    for filename in os.listdir(folderFeature1):
        f1 = np.load(folderFeature1 + '/' + filename)
        features1 = np.append(features1, [f1], axis=0)
    # print(features1)

    # tạo mảng 2 chiều lưu vị trí ảnh và giá trị cosine của nó với ảnh đầu vào
    compare = []
    for i in range(1,len(features)):
        tmp = cosine_similarity(features[i],features1[1])
        compare.append([tmp,i])
    # print("compare: ", compare)

    #chuyển sang numpy
    comparenp = np.array(compare)
    # print("comparenp: ", comparenp)

    # Xác định chỉ mục của vị trí đầu tiên trong mỗi hàng
    first_col_idx = np.argsort(comparenp[:, 0])
    # print("first_col_idx: ", first_col_idx)

    # Sắp xếp lại các hàng dựa trên chỉ mục đó
    sorted_arr = comparenp[first_col_idx]
    print("sort: ", sorted_arr)

    pos1 = int(sorted_arr[-1][1])#vị trí ảnh giống nhất trong dataset
    pos2 = int(sorted_arr[-2][1])
    pos3 = int(sorted_arr[-3][1])
    # print(pos)

    # print(str(float(sorted_arr[-1][0])), str(float(sorted_arr[-2][0])), str(float(sorted_arr[-3][0])))
    # if (float(sorted_arr[-1][0]) <0.7): print("không có ảnh nào phù hợp với input")
    # else:
    name1, name2, name3 = index_ouput(imgFolder, pos1, pos2, pos3)
    print(name1, name2, name3)
    print("ảnh giống với input nhất là "+ name1 + ".png với độ tương đồng là "+str(float(sorted_arr[-1][0])))#in vị trí

    