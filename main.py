import time
import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation


class TrackedArray():
    def __init__(self, arr):
        self.arr = np.copy(arr)
        self.reset()

    def reset(self):
        self.indices = []
        self.values = []
        self.access_type = []
        self.full_copies = []
        self.full_copies.append(np.copy(self.arr))

    def __track__(self, key, access_type):
        self.indices.append(key)
        self.values.append(self.arr[key])
        self.access_type.append(access_type)

    def __GetActivity__(self, idx=None):
        if isinstance(idx, type(None)):
            return [(i, op) for (i, op) in zip(self.indices, self.access_type)]
        else:
            return (self.indices[idx], self.access_type[idx])

    def __getitem__(self, key):
        self.__track__(key, "get")
        return self.arr.__getitem__(key)

    def __setitem__(self, key, value):
        self.arr.__setitem__(key, value)
        self.__track__(key, "set")

    def __len__(self):
        return self.arr.__len__()

#function for swapping elements in an array
def swap(arr, index1, index2):
    temp = arr[index1]
    arr[index1] = arr[index2]
    arr[index2] = temp

# insertion sort 
def insertion(arr):
    i = 1
    while i < len(arr):
        j = i
        while j > 0 and arr[j-1] > arr[j]:
            swap(arr, j, j-1)
            j = j - 1
        i = i + 1


plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 16
FPS = 60.0
N = 30

dataset = np.round(np.linspace(0, 1000, N), 0)
np.random.seed(8)
np.random.shuffle(dataset)


dataset = TrackedArray(dataset)

sorter = "Insertion Sort"

print(f"-----------{sorter}----------")

print("Before sorting: ")
print(dataset)

start = time.perf_counter()
insertion(dataset)
time_taken = time.perf_counter() - start

print("After sorting: ")
print(dataset)

print("Time taken (milliseconds): ", round(time_taken * 1000, 2), "ms")

fig, ax = plt.subplots()
container = ax.bar(np.arange(0, len(dataset), 1), dataset, align="edge", width=0.8)
ax.set_xlim([0, N])
ax.set(xlabel= "index", ylabel= "Value", title=f"{sorter}")


def update(frame):
    for (rectangle, height) in zip(container.patches, dataset.full_copies[frame]):
        rectangle.set_height(height)
        rectangle.set_color("#1f77b4")

    return (container)

ani = FuncAnimation(fig, update, frames=range(len(dataset.full_copies)),
                        blit=True, interval= 1000./FPS, repeat=False)

anim = ani
plt.show()
