from scipy import ndimage
import imageio.v2 as iv2
import numpy as np
import matplotlib.pyplot as plt

def SobelFilter(img, direction):
    if (direction == 'x'):
        Gx = np.array([[-1,0,+1], [-2,0,+2], [-1,0,+1]])
        Simage = ndimage.convolve(img, Gx)
    if (direction == 'y'):
        Gy = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
        Simage = ndimage.convolve(img, Gy)
    return Simage

def Normalise(img):
    # Nimg = img/np.max(img)
    Nimg = (img - np.mean(img))/(np.std(img))
    return Nimg

#loại bỏ các bounding boxes
def NonMaxSup(Gmag, Gmat):
    img = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0])-1):
        for j in range(1, int(Gmag.shape[1])-1):
            if((Gmat[i,j]>=-22.5 and Gmat[i,j]<=22.5) or (Gmat[i,j]<=-157.5 and Gmat[i,j]>=157.5)):
                if((Gmag[i,j]>Gmag[i,j+1]) and (Gmag[i,j]>Gmag[i, j-1])):
                    img[i,j] = Gmag[i,j]
                else: img[i,j] = 0

            if ((Gmat[i, j] >= 22.5 and Gmat[i, j] <= 67.5) or (Gmat[i, j] <= -112.5 and Gmat[i, j] >= -157.5)):
                if ((Gmag[i, j] > Gmag[i+1, j+1]) and (Gmag[i, j] > Gmag[i-1, j - 1])):
                    img[i, j] = Gmag[i, j]
                else:
                    img[i, j] = 0

            if ((Gmat[i, j] >= 67.5 and Gmat[i, j] <= 112.5) or (Gmat[i, j] <= -67.5 and Gmat[i, j] >= -112.5)):
                if ((Gmag[i, j] > Gmag[i + 1, j]) and (Gmag[i, j] > Gmag[i - 1, j])):
                    img[i, j] = Gmag[i, j]
                else:
                    img[i, j] = 0

            if ((Gmat[i, j] >= 122.5 and Gmat[i, j] <= 157.5) or (Gmat[i, j] <= -22.5 and Gmat[i, j] >= -67.5)):
                if ((Gmag[i, j] > Gmag[i+1, j-1]) and (Gmag[i, j] > Gmag[i-1, j + 1])):
                    img[i, j] = Gmag[i, j]
                else:
                    img[i, j] = 0
    return img

def DoThreshHyst(img):
    highThresholdRatio = 0.2
    lowThresholdRatio = 0.12
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])

    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    for i in range(1, h-1):
        for j in range(1, w-1):
            if(GSup[i,j]>highThreshold): GSup[i,j] = 1
            elif(GSup[i,j]<lowThreshold): GSup[i,j] = 0
            else:
                if((GSup[i-1,j-1] > highThreshold) or
                    (GSup[i-1,j] > highThreshold) or
                    (GSup[i-1,j+1] > highThreshold) or
                    (GSup[i,j-1] > highThreshold) or
                    (GSup[i,j+1] > highThreshold) or
                    (GSup[i+1,j-1] > highThreshold) or
                    (GSup[i+1,j] > highThreshold) or
                    (GSup[i+1,j+1] > highThreshold)): GSup[i,j] = 1
    GSup = (GSup == 1) * GSup
    return GSup



def run(img_inp):
    # img = iv2.imread("D:\\Hoc Ky 2_2023\\CSDL ĐPT\\BTL\\db\\in\\01_2.png")
    img = iv2.imread(img_inp)
    img = img.astype('int32')
    plt.imshow(img, cmap = plt.get_cmap('gray'))
    # plt.show()

    #khử nhiễu bằng bộ lọc gaussian
    img_gaussian = ndimage.gaussian_filter(img, sigma=1.4)
    plt.imshow(img_gaussian, cmap=plt.get_cmap('gray'))
    # plt.show()

    #đạo hàm theo hướng x và y (Gradient, bộ lọc Sobel)
    gx = SobelFilter(img_gaussian, 'x')
    gx = Normalise(gx)
    gy = SobelFilter(img_gaussian, 'y')
    gy = Normalise(gy)
    plt.imshow(gx, cmap=plt.get_cmap('gray'))
    # plt.show()
    plt.imshow(gy, cmap=plt.get_cmap('gray'))
    # plt.show()

    Mag = np.hypot(gx, gy)
    plt.imshow(Mag, cmap=plt.get_cmap('gray'))
    # plt.show()

    #chuyển sang độ
    Gmat = np.degrees(np.arctan(gy, gx))
    # print(Gmat)

    # img_NMS = NonMaxSup(Mag, Gmat)
    # img_NMS = Normalise(img_NMS)
    # plt.imshow(img_NMS, cmap=plt.get_cmap('gray'))
    # plt.show()

    final_img = DoThreshHyst(Gmat)
    plt.imshow(final_img, cmap=plt.get_cmap('gray'))
    # plt.show()
    return final_img, gx, gy

# run(1)