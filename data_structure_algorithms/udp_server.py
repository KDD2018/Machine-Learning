import socket


def main():
    # 创建套接字
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定IP和port
    local_addr = ("", 8190)
    server.bind(local_addr)
    # 接受发送消息
    while True:
        msg, client_addr = server.recvfrom(1024)
        if msg:
            print("【%s】, Recv from client: %s" % (msg.decode('utf-8'), client_addr))
            server.sendto(("接受到消息：[{0}]".format(msg.decode())).encode(), client_addr)
        else:
            break
    # 关闭套接字
    server.close()


if __name__ == '__main__':
    main()