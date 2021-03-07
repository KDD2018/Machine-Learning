def bucket_sort(s, length):
    """桶排序"""
    min_num = min(s)
    max_num = max(s)
    # 桶的大小
    bucket_range = (max_num-min_num) / length
    # 桶数组
    count_list = [ [] for i in range(length + 1)]
    # 向桶数组填数
    for i in s:
        count_list[int((i-min_num)//bucket_range)].append(i)
    s.clear()
    # 回填，这里桶内部排序直接调用了sorted
    for i in count_list:
        for j in sorted(i):
            s.append(j)


def bubbleSort(arr, length):
    for i in range(1, length):
        for j in range(0, length-i):
            if arr[j] < arr[j+1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def insertionSort(arr, length):
    for i in range(length):
        preIndex = i-1
        current = arr[i]
        while preIndex >= 0 and arr[preIndex] < current:
            arr[preIndex+1] = arr[preIndex]
            preIndex-=1
        arr[preIndex+1] = current
    return arr


if __name__ == '__main__':
    length = int(input())
    a = list(map(int, input().strip().split()))
    # bucket_sort(a, length)
    # bubbleSort(a, length)
    insertionSort(a, length)
    print(a)
