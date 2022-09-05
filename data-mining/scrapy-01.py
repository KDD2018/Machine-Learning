from urllib.request import urlopen

url = "https://www.bilibili.com/"
response = urlopen(url)
# print(response.read().decode('utf-8'))
with open('baidu.html', 'w') as f:
    f.write(response.read().decode('utf-8'))