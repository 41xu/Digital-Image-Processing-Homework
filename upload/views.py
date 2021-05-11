from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import os
import datetime
import time
import cv2
import matplotlib.pyplot as plt
from scipy import fftpack
from PIL import Image


@csrf_exempt
def upload(request):
    if request.method == 'GET':
        return render(request, 'upload.html')
    if request.method == 'POST':
        img1, img2, img3 = False, False, False
        pth, res_pth, pth1, pth2 = "/", "/", "/", "/"
        if request.POST.get("image"):
            image = request.FILES.get('image1')
            pth = './images/' + image.name
            with open(pth, 'wb') as f:
                for content in image.chunks():
                    f.write(content)
            img1 = True
            if request.POST.get('image') == 'gray':
                res_pth = histogram_equal_gray(pth)
            elif request.POST.get('image') == 'rgb':
                res_pth = histogram_equal_rgb(pth)
        elif request.FILES.get('image2'):
            image = request.FILES.get("image2")
            pth = './images/' + image.name
            with open(pth, 'wb') as f:
                for content in image.chunks():
                    f.write(content)
            if image.name.split('.')[-1] in ['tiff', 'tif']:
                tmp = Image.open(pth)
                pth = './images/' + image.name.split('.')[0] + '.jpg'
                print(pth)
                tmp.save(pth, 'JPEG')
            img2 = True
            res_pth = laplacian(pth)
        elif request.FILES.get("image_match") and request.FILES.get("image_template"):
            template = request.FILES.get("image_template")
            image = request.FILES.get("image_match")
            pth1 = './images/' + template.name
            pth2 = './images/' + image.name
            with open(pth1, 'wb') as f:
                for content in template.chunks():
                    f.write(content)
            with open(pth2, 'wb') as f:
                for content in image.chunks():
                    f.write(content)
            img3 = True
            pth = pth2
            res_pth = fftmatch(pth1, pth2)
        template = loader.get_template('upload.html')
        context = dict(
            img1 = img1,
            img2 = img2,
            img3 = img3,
            origin_img_pth=pth[1:],
            res_img_pth=res_pth[1:],
            template_pth = pth1[1:],
        )
        return HttpResponse(template.render(context, request))
    return render('upload.html')


def histogram_equal_gray(pth):
    img = cv2.imread(pth)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", img_gray)
    (w, h) = img_gray.shape
    # p1 = plt.hist(img_gray.reshape(img_gray.size, 1))
    # plt.show()
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
        c[i] = c[i - 1] + p[i]
    des = np.zeros((w, h), dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            des[x][y] = 255 * c[img_gray[x][y]]
    cv2.imwrite(pth[:-4] + "_res.png", des)
    res_pth = pth[:-4] + "_res.png"
    # cv2.imshow("res", des)
    # cv2.waitKey(0)
    # p2 = plt.hist(des.reshape(des.size, 1))
    # plt.show()
    # return des
    return res_pth


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
        c[i] = c[i - 1] + p[i]
    des = np.zeros((w, h), dtype=np.uint8)
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
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("res", res)
    # cv2.waitKey(0)
    cv2.imwrite(pth[:-4] + "_res.png", res)
    res_pth = pth[:-4] + "_res.png"
    return res_pth


def laplacian(pth, ksize=3):
    img = cv2.imread(pth, 0)  # gray image
    h, w = img.shape
    pad = ksize // 2
    out = np.zeros((h + pad * 2, w + pad * 2), dtype=np.float)
    out[pad: pad + h, pad: pad + w] = img.copy().astype(np.float)
    tmp = out.copy()
    # k = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
    k = [[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]
    for y in range(h):
        for x in range(w):
            out[pad + y, pad + x] = -1 * np.sum(k * (tmp[y: y + ksize, x: x + ksize])) + tmp[pad + y, pad + x]
    out = np.clip(out, 0, 255)
    out = out[pad:pad + h, pad: pad + w].astype(np.uint8)
    cv2.imwrite(pth[:-4] + "_lapres.png", out)
    res_pth = pth[:-4] + "_lapres.png"
    return res_pth


def fftmatch(wind, trg):
    wind_origin = cv2.imread(wind)
    trg_origin = cv2.imread(trg)
    window_name, trg_name = wind, trg
    wind = cv2.cvtColor(wind_origin, cv2.COLOR_BGR2GRAY)
    trg = cv2.cvtColor(trg_origin, cv2.COLOR_BGR2GRAY)
    wh, ww = wind.shape
    th, tw = trg.shape
    wfft = fftpack.fft2(wind, shape=(th, tw))
    tfft = fftpack.fft2(trg)
    c = np.fft.ifft2(np.multiply(wfft,  tfft.conj())) / np.abs(np.multiply(wfft, tfft.conj()))
    c = c.real
    res = np.where(c == np.amax(c))
    maxcoor = list(zip(res[0], res[1]))[0]
    start = (maxcoor[1] - wh // 2, maxcoor[0] - ww // 2)
    end = (maxcoor[1] + wh // 2, maxcoor[0] + ww // 2)
    # blue color
    color = (255, 0, 0)
    thickness = 2
    img = cv2.rectangle(trg_origin, start, end, color, thickness)
    # cv2.imshow("matchting", img)
    # cv2.waitKey(0)
    cv2.imwrite(trg_name[:-4] + "_matching.png", img)
    res_pth = trg_name[:-4] + "_matching.png"
    return res_pth