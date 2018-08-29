# coding = utf-8
import socket
import threading
import csv
from sklearn import neighbors
import cv2


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


def knn_train():
    global knn
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_data_set, train_target_set)


def knn_prediction():
    # train_target_set, train_data_set = load_data_set()
    test_data_set = []

    image = cv2.imread('hello.jpg')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resize = cv2.resize(image_gray, (28, 28))
    for i in range(0, 28):
        for j in range(0, 28):
            test_data_set.append(image_resize[i][j])

    prediction = knn.predict([test_data_set])
    print "Prediction ok"
    print prediction[0]
    return prediction[0]


def tcplink(sock, addr):
    print "Connection from %s:%s" % addr
    f = open('hello.jpg', 'wb')
    while True:
        data = sock.recv(1024)
        if data == 'exit' or not data:
            res = knn_prediction()
            sock.send(str(res))
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
