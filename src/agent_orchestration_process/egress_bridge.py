"""Run a command inside an AOP network namespace with its egress bridge.

Usage: egress_bridge.py SOCKET PROXY_PORT [HOST:PORT ...] -- COMMAND...

Inside the namespace, 127.0.0.1:PROXY_PORT reaches the controller's CONNECT
proxy through SOCKET, and each HOST:PORT (a loopback endpoint) is forwarded to
the same endpoint outside. The controller decides what is allowed, so this
script holds no policy. It forwards stop signals to the command and exits
with the command's status; bubblewrap's own process 1 collects orphans.
"""

import signal
import socket
import subprocess
import sys
import threading


def pipe(left, right):
    def copy(source, target):
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                target.sendall(data)
        except OSError:
            pass
        finally:
            try:
                target.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    other = threading.Thread(target=copy, args=(right, left), daemon=True)
    other.start()
    copy(left, right)
    other.join()


def upstream(socket_path, connect_to=None):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_path)
    if connect_to is not None:
        connection.sendall(f"CONNECT {connect_to} HTTP/1.1\r\n\r\n".encode())
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = connection.recv(4096)
            if not chunk:
                break
            response += chunk
        if not response.startswith(b"HTTP/1.1 200"):
            connection.close()
            raise OSError(f"proxy refused {connect_to}")
    return connection


def serve(port, socket_path, connect_to=None):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(64)

    def handle(client):
        with client:
            try:
                with upstream(socket_path, connect_to) as remote:
                    pipe(client, remote)
            except OSError:
                pass

    def accept():
        while True:
            client, _ = listener.accept()
            threading.Thread(target=handle, args=(client,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()


def main(argv):
    separator = argv.index("--")
    socket_path, proxy_port, *forwards = argv[:separator]
    command = argv[separator + 1 :]
    serve(int(proxy_port), socket_path)
    for item in forwards:
        serve(int(item.rpartition(":")[2]), socket_path, connect_to=item)
    child = subprocess.Popen(command)
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, lambda signum, frame: child.send_signal(signum))
    code = child.wait()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
