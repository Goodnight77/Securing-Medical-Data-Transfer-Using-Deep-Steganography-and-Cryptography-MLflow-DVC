import socket
HOST = 'localhost'
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST , PORT ))
    data= s.recv(1024)
    #save image or display it