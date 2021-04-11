import socket


def main():
    # 创建套接字
    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定本地信息‘
    tcp_server.bind(('', 8192))

    # 设置监听模式
    tcp_server.listen(128)
    print('监听中......')

    while True:
        # 等待客户端的链接
        client_socket, client_addr = tcp_server.accept()
        if client_socket:
            print(f'客户端{client_addr}已链接...')

            while True:
                # 接受信息
                recv_data = client_socket.recv(1024)
                if recv_data:
                    print(f"客户端：{recv_data.decode('utf-8')}")

                    # 回送数据给客户端
                    client_socket.send('收到----OK-----'.encode('utf-8'))
                else:
                    break
            # 关闭套接字
            client_socket.close()
        else:
            break
    tcp_server.close()


if __name__ == '__main__':
    main()

