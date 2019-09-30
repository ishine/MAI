import numpy as np

data = np.arange(1,28,1).reshape(3,3,3)

indices = np.array([[0,2],[1,2]])


print("Multi dimensions indices, axis is 0===============================")
y = np.take(data, indices, axis=0)
print(y.shape)
print(y)

print("Multi dimensions indices, axis is 1===============================")
y = np.take(data, indices, axis=1)
print(y.shape)
print(y)

print("Multi dimensions indices, axis is 2===============================")
y = np.take(data, indices, axis=2)
print(y.shape)
print(y)
