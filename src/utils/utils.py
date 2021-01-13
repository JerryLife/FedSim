
def get_split_points(array, size):
    assert size > 1

    prev = array[0]
    split_points = [0]
    for i in range(1, size):
        if prev != array[i]:
            prev = array[i]
            split_points.append(i)

    split_points.append(size)
    return split_points


def move_item_to_end_(arr, items):
    for item in items:
        arr.insert(len(arr), arr.pop(arr.index(item)))


def move_item_to_start_(arr, items):
    for item in items[::-1]:
        arr.insert(0, arr.pop(arr.index(item)))
