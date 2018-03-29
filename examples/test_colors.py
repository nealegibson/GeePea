#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

data = np.arange(9, dtype=float).reshape((3, 3)) / 8.

color = np.array([1, 0, 0, 1], dtype=float)

array = color * np.ones(data.shape + (4,))
array[:, :, 3] *= data

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(array, vmin=0, vmax=1, interpolation='nearest',origin='upper')
for i in range(3):
    for j in range(3):
        ax.scatter(i, j, s=500, facecolor='white', edgecolor='white', alpha=1.0)
        ax.scatter(i, j, s=400, facecolor=array[j, i], edgecolor='white')
#fig.savefig('colors.pdf')

