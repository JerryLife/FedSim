
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
