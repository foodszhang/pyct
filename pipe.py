from multiprocessing.connection import Listener
import traceback

def echo_client(conn):
    try:
        while True:
            msg = conn.recv()
            print('!!!recv',msg)
    except EOFError:
        print('Connection closed')

def send_msg(conn, msg):
    conn.send(msg)

def detector_server(address, authkey, callback=echo_client):
    serv = Listener(address, authkey=authkey)
    client = serv.accept()

    callback(client)


if __name__ == '__main__':
    detector_server(r'\\.\pipe\detectResult', b'ctRestruct')
