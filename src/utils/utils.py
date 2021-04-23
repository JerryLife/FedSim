from nltk.metrics.distance import edit_distance
import faiss


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


def scaled_edit_distance(a: str, b: str):
    return edit_distance(a, b) / max(len(a), len(b))


def custom_index_cpu_to_gpu_multiple(resources, index, co=None, gpu_nos=None):
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if gpu_nos is None:
        gpu_nos = range(len(resources))
    for i, res in zip(gpu_nos, resources):
        vdev.push_back(i)
        vres.push_back(res)
    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
    index.referenced_objects = resources
    return index

