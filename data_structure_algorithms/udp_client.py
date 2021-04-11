import socket


def main():
    # 创建套接字
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 目标ip和port
    ADDR = ("192.168.0.115", 8190)
    # 发送内容
    while True:
        msg = input("请输入信息：")
        if not msg or msg == 'exit':
            break
        # 发送
        client.sendto(msg.encode("utf-8"), ADDR)
        data, server_addr = client.recvfrom(1024)
        print("%s, 消息来自服务端:%s" % (data.decode('utf-8'), server_addr))
    # 关闭套接字
    client.close()


if __name__ == '__main__':
    main()

