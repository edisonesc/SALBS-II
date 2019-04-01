import cv2
import numpy as np
import socket
import sys
import pickle
import struct

cap = cv2.VideoCapture(0)
def setupSocket():
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(('127.0.0.1', 8083))
    return clientsocket
clientsocket = setupSocket()

# reply = clientsocket.recv(1024)
# print(reply.decode('utf-8'))

while True:
    data = clientsocket.recv(1024).decode('utf-8')  # receive response
    print(data)
    ret, frame = cap.read()

    data = pickle.dumps(frame)
    clientsocket.sendall(struct.pack("L", len(data)) + data)




