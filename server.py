import socket
import sys
import cv2
import pickle
import numpy as np
import struct
import pyzbar.pyzbar as pyzbar


# HOST = '192.168.1.4'
QR_SCAN, FIND_MATCH_SHELF, RETURN_BOOK, GO_BACK = False, False, False, False

HOST = '127.0.0.1'
# HOST = '192.168.1.4'
# PORT = 4957
PORT = 8083

def setupServer():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')
    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')
    return s
s = setupServer()
def setupConnection():
    conn, addr = s.accept()
    print("Connected to: {0} : {1}".format(addr[0], addr[1]))
    return conn


def main():
    conn = setupConnection()
    data = b''
    conn.sendall(str.encode("FROM SEVER"))

    payload_size = struct.calcsize("L")
    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        qr_display(frame, decode(frame)[0])
        if(decode(frame)[1] == 1):
            QR_SCAN = True
            # d = get(str(decode(frame)[1]))
            # conn.sendall(str.encode(d))
        #     data = decode(frame)[2] # get qr code data


        print(frame.size)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)


def qr_scan_book(frame):
    pass


def qr_display(frame, decodedObjects):
    # Loop over all decoded objects
    for decodedObject in decodedObjects:
        points = decodedObject.polygon
        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points
        n = len(hull)
        for j in range(0, n):
            cv2.line(frame, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

def decode(frame):
    while True:
        decodedObjects = pyzbar.decode(frame)
        for obj in decodedObjects:
            print('Type : ', obj.type)
            print('Data : ', obj.data, '\n')



        return decodedObjects, len(decodedObjects)



main()



cv2.destroyAllWindows()
