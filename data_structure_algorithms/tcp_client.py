import socket


def main():
    # 创建tcp套接字, tcp对应socket.SOCK_STREAM
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 链接服务器
    server_ip = input('请输入服务器IP：')
    server_port = int(input('请输入服务器端口：'))
    server_addr = (server_ip, server_port)
    tcp_socket.connect(server_addr)

    # 发送数据/接收数据
    send_msg = input('请输入消息：')
    tcp_socket.send(send_msg.encode('utf-8'))

    # 关闭套接字
    tcp_socket.close()


if __name__ == '__main__':
    main()

