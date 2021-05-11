import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


def laplacian_filter(img, ksize=3):
    """
    input img: cv2.imread(pth, 0), 读入的是灰度图
    """
    h, w = img.shape
    pad = ksize // 2
    out = np.zeros((h + pad * 2, w + pad * 2), dtype=np.float)
    out[pad: pad + h, pad: pad + w] = img.copy().astype(np.float)
    tmp = out.copy()
    k = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
    for y in range(h):
        for x in range(w):
            out[pad+y, pad+x] = -1 * np.sum(k * (tmp[y: y+ksize, x: x+ksize]))
    out = np.clip(out, 0, 255)
    out = out[pad:pad+h, pad: pad+w].astype(np.uint8)
    return out


def laplacian(img, ksize=3):
    h, w = img.shape
    pad = ksize // 2
    out = np.zeros((h + pad * 2, w + pad * 2), dtype=np.float)
    out[pad: pad + h, pad: pad + w] = img.copy().astype(np.float)
    tmp = out.copy()
    k = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
    for y in range(h):
        for x in range(w):
            out[pad+y, pad+x] = -1 * np.sum(k * (tmp[y: y+ksize, x: x+ksize])) + tmp[pad+y, pad+x]
    out = np.clip(out, 0, 255)
    out = out[pad:pad+h, pad: pad+w].astype(np.uint8)
    return out


def histogram_equal_gray(pth):
    img = cv2.imread(pth)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray", img_gray)
    (w, h) = img_gray.shape
    p1 = plt.hist(img_gray.reshape(img_gray.size, 1))
    plt.show()
    n = np.zeros((256), dtype=np.float)
    p = np.zeros((256), dtype=np.float)
    c = np.zeros((256), dtype=np.float)
    for i in range(w):
        for j in range(h):
            n[img_gray[i][j]] += 1
    for i in range(256):
        p[i] = n[i] / float(img_gray.size)
    c[0] = p[0]
    for i in range(1, 256):
        c[i] = c[i-1] + p[i]
    des = np.zeros((w,h), dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            des[x][y] = 255 * c[img_gray[x][y]]

    cv2.imshow("res", des)
    cv2.waitKey(0)
    p2 = plt.hist(des.reshape(des.size, 1))
    plt.show()
    return des


def test(img):
    """
    img: GRAY
    """
    (w, h) = img.shape
    # p1 = plt.hist(img.reshape(img.size, 1))
    # plt.show()
    n = np.zeros((256), dtype=np.float)
    p = np.zeros((256), dtype=np.float)
    c = np.zeros((256), dtype=np.float)
    for i in range(w):
        for j in range(h):
            n[img[i][j]] += 1
    for i in range(256):
        p[i] = n[i] / float(img.size)
    c[0] = p[0]
    for i in range(1, 256):
        c[i] = c[i-1] + p[i]
    des = np.zeros((w,h), dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            des[x][y] = 255 * c[img[x][y]]

    # cv2.imshow("res", des)
    # cv2.waitKey(0)
    # p2 = plt.hist(des.reshape(des.size, 1))
    # plt.show()
    return des



def histogram_equal_rgb(pth):
    """
    彩色图像先做split切分三通道，然后分别进行均衡化，最后merge()
    """
    img = cv2.imread(pth)
    (b, g, r) = cv2.split(img)
    # print(b,g,r)
    # bh = histogram_equal_gray(b)
    bh = test(b)
    gh = test(g)
    rh = test(r)
    res = cv2.merge((bh, gh, rh))
    print("----")
    print(res.shape)
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("res", res)
    # cv2.waitKey(0)


def rgb2hsi(pth):
    img = cv2.imread(pth, 1)
    h, w =img.shape
    B, G, R = cv2.split(img)
    B /= 255.
    G /= 255.
    R /= 255.
    tmp = img.copy()
    H, S, I = cv2.split(tmp)
    for i in range(h):
        for j in range(w):
            num = 0.5 * ((R[i, j] - G[i, j]) + (R[i, j] - B[i, j]))
            den = np.sqrt((R[i, j] - G[i, j]) ** 2 + (R[i, j] - B[i, j]) * (G[i, j] - B[i, j]))
            theta = np.float(np.arccos(num / den))

            if den == 0:
                H = 0
            elif B[i, j] <= G[i, j]:
                H = theta
            else:
                H = 2 * np.pi - theta

            minRGB = min(min(B[i, j], G[i, j]), R[i, j])
            sum = B[i, j] + G[i, j] + R[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * minRGB / sum
            H /= 2 * np.pi
            I = sum / 3.

            # 计算直方图扩充
            tmp[i, j, 0] = H * 255
            tmp[i, j, 1] = S * 255
            tmp[i, j, 2] = I * 255
            # # [0度, 360度], [0, 100], [0, 255]
            # tmp[i, j, 0] = H * 360
            # tmp[i, j, 1] = S * 100
            # tmp[i, j, 2] = I * 255
    return tmp



def histogram_equal_hsi(pth):
    img = rgb2hsi(pth)
    h, s, i = cv2.split(img)
    hh, sh, ih = test(h), test(s), test(i)
    res = cv2.merge((hh, sh, ih))
    cv2.imshow("tset", res)
    cv2.waitKey(0)

def fftmatch(wind, trg):
    wind_origin = cv2.imread(wind)
    trg_origin = cv2.imread(trg)
    wind = cv2.cvtColor(wind_origin, cv2.COLOR_BGR2GRAY)
    trg = cv2.cvtColor(trg_origin, cv2.COLOR_BGR2GRAY)
    wh, ww = wind.shape
    th, tw = trg.shape
    wfft = fftpack.fft2(wind, shape=(th, tw))
    tfft = fftpack.fft2(trg)
    c = np.fft.ifft2(np.multiply(wfft, tfft.conj())) / np.abs(np.multiply(wfft, tfft.conj()))
    c = c.real
    res = np.where(c == np.amax(c))
    maxcoor = list(zip(res[0], res[1]))[0]
    print(maxcoor)
    start = (maxcoor[1], maxcoor[0])

    end = (maxcoor[1] + th, maxcoor[0] + tw)
    # blue color
    color = (255, 0, 0)
    thickness = 2
    img = cv2.rectangle(trg_origin, start, end, color, thickness)
    cv2.imshow("matchting", img)
    cv2.waitKey(0)





if __name__ == '__main__':
    ####################
    # histogram equal gray test
    ####################
    # pth = '../2021-ImagesSet/histeq1.jpg'
    # histogram_equal_gray(pth)
    ####################
    # histogram equal rgb test
    ####################
    pthrgb = '../2021-ImagesSet/histeqColor.jpg'
    img = cv2.imread('../2021-ImagesSet/histeqColor.jpg')
    histogram_equal_rgb(pthrgb)
    # histogram_equal_hsi(pthrgb)
    ####################
    # Laplacian test
    ####################
    # img = cv2.imread("../2021-ImagesSet/moon.tif", 0).astype(np.float) # cv2.imread(pth, 0:灰度图，1:rgb)
    # out = laplacian(img)
    # filter = laplacian_filter(img)
    # cv2.imshow("img", cv2.imread('../2021-ImagesSet/moon.tif', 1))
    # cv2.imshow("filter + img", out)
    # cv2.imshow("filter", filter)
    # cv2.waitKey(0)
    ####################
    # fftmacth
    ####################
    # wind = '../2021-ImagesSet/match_window.jpg'
    # target = '../2021-ImagesSet/match.jpg'
    # fftmatch(wind, target)