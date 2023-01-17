import numpy as np

o = np.random.randint(0, 2, 20)

ap = o + np.roll(o, 1, 0)
print(ap[ap==2])
print(ap)
print(ap.astype(bool).astype(int))