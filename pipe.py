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

def detector_server(
    address, authkey, callback=echo_client, ready_event=None, listener_holder=None
):
    serv = None
    client = None
    try:
        serv = Listener(address, authkey=authkey)
        if listener_holder is not None:
            listener_holder["listener"] = serv
        if ready_event is not None:
            ready_event.set()
        client = serv.accept()
        callback(client)
    finally:
        try:
            if client is not None:
                client.close()
        except Exception:
            pass
        try:
            if serv is not None:
                serv.close()
        except Exception:
            pass


if __name__ == '__main__':
    detector_server(r'\\.\pipe\detectResult', b'ctRestruct')
