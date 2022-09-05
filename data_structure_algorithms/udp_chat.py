import socket
import threading


def send_msg(udp_socket, dest_ip, dest_port):
    """发送消息"""
    while True:
        send_data = input('\n请输入要发送的消息：')
        udp_socket.sendto(send_data.encode('utf-8'), (dest_ip, dest_port))


def recv_msg(udp_socket):
    """接收消息"""
    while True:
        recv_data = udp_socket.recvfrom(1024)
        print(f"\n消息来自{recv_data[1].decode('utf-8')}：{recv_data[0]}")


def main():
    # 1、 创建套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 2. 绑定本地信息
    udp_socket.bind(("", 8190))

    # 3. 获取对方IP和Port
    dest_ip = input('\n请输入对方IP：')
    dest_port = int(input("\n请输入对方Port:"))

    # 4. 创建子线程，执行接发消息
    t_recv = threading.Thread(target=recv_msg, args=(udp_socket,))
    t_send = threading.Thread(target=send_msg, args=(udp_socket, dest_ip, dest_port))
    t_recv.start()
    t_send.start()


if __name__ == '__main__':
    main()
