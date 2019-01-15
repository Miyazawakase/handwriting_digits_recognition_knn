# coding:UTF-8
import socket
import threading
import csv
from sklearn import neighbors
import cv2
from time import ctime
import numpy as np


train_target_set = []
train_data_set = []
knn = None


def load_data_set():
    with open('train.csv') as csv_train:
        csv_train = csv.reader(csv_train)
        train_set = list(csv_train)

    for i in range(1, len(train_set)):
        temp = []
        train_target_set.append(int(train_set[i][0]))
        for j in range(1, 785):
            train_set[i][j] = int(train_set[i][j])
            temp.append(train_set[i][j])
        train_data_set.append(temp)

    # return train_target_set, train_data_set


def centralization(image_gray):
    dst = np.zeros((280, 280))
    for i in range(280):
        for j in range(280):
            dst[i][j] = 255
    # dst.dtype = 'uint8'
    # (r, c) = image_gray.shape

    # 先把原图缩小成一个长方形
    image_gray = cv2.resize(image_gray, (80, 200))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # for i in range(0, 1):
    #     image_gray = cv2.erode(image_gray, kernel)
    # 再将这个长方形填入28*28的全0矩阵中
    idk1 = 0
    idk2 = 0
    for i in range(40, 240):
        for j in range(100, 180):
            dst[i][j] = image_gray[idk1][idk2]
            if idk2 == 79:
                idk1 += 1
                idk2 = 0
            else:
                idk2 += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for i in range(0, 3):
        dst = cv2.erode(dst, kernel)
    dst = cv2.resize(dst, (28, 28))
    # cv2.imshow("DST", dst)
    # cv2.imshow("Gray", image_gray)
    # cv2.waitKey(0)

    return dst


def anti_binarization(image_gray):
    (r, c) = image_gray.shape
    for idk1 in range(r):
        for idk2 in range(c):
            if image_gray[idk1][idk2] > 125:
                image_gray[idk1][idk2] = 0
            else:
                image_gray[idk1][idk2] = 255

    return image_gray


def knn_train():
    global knn
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_data_set, train_target_set)


def knn_prediction(image_gray):
    # train_target_set, train_data_set = load_data_set()
    test_data_set = []

    # image = cv2.imread('hello.jpg')
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resize = centralization(image_gray)
    image_resize = anti_binarization(image_resize)
    # cv2.imshow("Res", image_resize)
    # cv2.waitKey(0)

    for i in range(0, 28):
        for j in range(0, 28):
            test_data_set.append(image_resize[i][j])

    # cv2.imshow("Hello", image_resize)
    # cv2.waitKey(0)

    prediction = knn.predict([test_data_set])
    print "Prediction ok"
    print prediction[0]
    return prediction[0]


def divPic(image_gray):
    (r, c) = image_gray.shape
    print ("r:", r, " c:", c)
    # 对hello.jpg进行分割处理
    res = []
    # 记录每个数字的左右边界
    widthPair = []
    left = right = -1
    for idk2 in range(c):
        if 0 in image_gray[:, idk2] and left == -1:
            left = idk2
        if 0 not in image_gray[:, idk2] and left != -1:
            right = idk2
            widthPair.append((left, right))
            left = right = -1
    # 确定每个数字的上下边界
    top = bottom = -1
    for pair in widthPair:
        for i in range(r):
            if 0 in image_gray[i, pair[0]: pair[1]] and top == -1:
                top = i
            if 0 not in image_gray[i, pair[0]: pair[1]] and top != -1:
                bottom = i
                res.append(image_gray[top: bottom, pair[0]: pair[1]])
                top = -1
                bottom = -1

    # 将分割完毕的每一个数字显示出来
    # for i, element in enumerate(res):
    #     cv2.imshow("Number" + str(i), element)
    #     cv2.waitKey(0)

    return res


def binarization(image_gray):
    # 二值化
    (r, c) = image_gray.shape
    # print ("r:", r, " c:", c)
    for idk1 in range(r):
        for idk2 in range(c):
            if image_gray[idk1][idk2] > 125:
                image_gray[idk1][idk2] = 255
            else:
                image_gray[idk1][idk2] = 0

    # cv2.imshow("Gray", image_gray)
    # cv2.waitKey(0)

    return image_gray


def tcplink(sock, addr):
    print "Connection from %s:%s" % addr
    f = open('hello.jpg', 'wb')
    while True:
        data = sock.recv(1024)
        if data == 'exit' or not data:
            print ("Recognition!")
            # 图片接收结束，读取图片并灰度化、二值化
            image = cv2.imread('hello.jpg')
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print (image_gray.shape)
            image_gray = binarization(image_gray)
            # 这个divNum是分割后的每个数字的集合，是numpy数组的集合
            divNum = divPic(image_gray)
            # 对每个数字进行识别，把结果字符串发送回去
            res = ""
            for i, element in enumerate(divNum):
                # cv2.imwrite("num" + str(i) + ".jpg", element)
                prediction = knn_prediction(element)
                res += str(prediction)
            print res
            sock.send(ctime() + " " + res)
            break
        f.write(data)
    sock.close()
    print "Connection from %s:%s closed." % addr


def main():
    load_data_set()
    print "Data ready"
    knn_train()
    print "Training ready"

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 9999))
    s.listen(5)
    print "Waiting for connection......"
    while True:
        sock, addr = s.accept()
        t = threading.Thread(target=tcplink, args=(sock, addr))
        t.start()


if __name__ == '__main__':
    main()
