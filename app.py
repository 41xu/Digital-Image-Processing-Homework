import cv2
import numpy as np
import matplotlib.pyplot as plt


def laplacian(img, kernel_size=3):
    (h, w) = img.shape
    pad = kernel_size // 2
    out = np.zeros((h + pad * 2, w + pad*2), dtype=np.float)
    out[pad:pad+h, pad:pad+w] = img.copy().astype(np.float)
    tmp = out.copy()
    K = [[0.,  1., 0.], [1., -4., 1.], [0., 1., 0.]]
    for y in range(h):
        for x in range(w):
            out[pad+y, pad+x] = np.sum(K*tmp[y: y+kernel_size, x:x+kernel_size])
    out = np.clip(out, 0, 255)
    out = out[pad: pad+h, pad+w].astype(np.uint8)
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
    p1 = plt.hist(img.reshape(img.size, 1))
    plt.show()
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

    cv2.imshow("res", des)
    cv2.waitKey(0)
    p2 = plt.hist(des.reshape(des.size, 1))
    plt.show()
    return des



def histogram_equal_rgb(pth):
    """
    彩色图像先做split切分三通道，然后分别进行均衡化，最后merge()
    """
    img = cv2.imread(pth)
    (b, g, r) = cv2.split(img)
    print(b,g,r)
    # bh = histogram_equal_gray(b)
    bh = test(b)
    gh = test(g)
    rh = test(r)
    # gh = histogram_equal_gray(g)
    #
    # rh = histogram_equal_gray(r)
    res = cv2.merge((bh, gh, rh))
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    cv2.imshow("res", res)
    cv2.waitKey(0)


def histogram_equal_hsi(pth):
    pass



if __name__ == '__main__':
    # pth = '../2021-ImagesSet/histeq1.jpg'
    # histogram_equal_gray(pth)
    pthrgb = '../2021-ImagesSet/histeqColor.jpg'
    # img = cv2.imread('../2021-ImagesSet/histeqColor.jpg')
    # print(img)
    histogram_equal_rgb(pthrgb)
