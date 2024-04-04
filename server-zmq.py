import numpy as np

import tensorcom

# python server-zmq.py

server = tensorcom.Connection(multipart=False)
server.connect("zserver://127.0.0.1:7888")
# server.connect("zclient+ipc://mypath")

while True:
    data = server.recv()
    print("{}".format(data[:16]))

    # data = np.asarray([1,1371,592], dtype=np.int32)
    # data = np.random.randint(0, 32832, size=(2048,), dtype=np.uint16)
    data = np.random.randint(0, 32000, size=(4,), dtype=np.uint16)
    server.send([data,])
