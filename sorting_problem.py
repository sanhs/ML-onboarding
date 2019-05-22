N = 5; M = 3
arr = [
[10, 2, 5],
[7, 1, 0],
[9, 9, 9],
[1, 23, 12],
[6, 5, 9]
]
k = 1


def merge(arr, l, r, m):
    left_arr = []; right_arr = []
    
    for x in range(l, m+1):
        left_arr.append(arr[x])
    
    for x in range(m+1, r+1):
        right_arr.append(arr[x])
    
    i = 0; j = 0; k = l
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1
    
    while i < len(left_arr):
        arr[k] = left_arr[i]
        i += 1
        k += 1
    
    while j < len(right_arr):
        arr[k] = right_arr[j]
        j += 1
        k += 1


def merge_sort(arr, l, r):
    if l < r:
        m = int((l+r)/2)
        merge_sort(arr, l, m)
        merge_sort(arr, m+1, r)
        merge(arr, l, r, m)
    return arr


def sorted(arr, k):
    if len(arr) < 1 or len(arr[0]) < 1:
        return arr
    if len(arr[0]) == 1:
        return merge_sort(arr, 0, len(arr)-1)
    
    row_map = {}
    k_ele = []
    for row in arr:
        if row_map.__contains__(row[k]):
            row_map[row[k]].append(row)
        else:
            row_map[row[k]] = [row]
            k_ele.append(row[k])
    
    sorted_k_ele = merge_sort(k_ele, 0, len(k_ele)-1)
    
    out_arr = []
    for ele in k_ele:
        out_arr.extend(row_map[ele])
    
    return out_arr



#out_arr = merge_sort([5, 4, 3, 2, 1], 0, 4)
out_arr = sorted(arr, k=1)
print(out_arr)