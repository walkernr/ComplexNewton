# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:32:51 2014

@author: Nick Walker
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as an
import itertools as it
import random as r
import sys

d = 10
n = 1000

f = np.random.randint(10, size = r.randint(2, 10))

class newton:

    def __init__(self, func, xmin, xmax, ymin, ymax, xres, yres):
        self._x = np.linspace(xmin, xmax, xres)
        self._y = np.linspace(ymin, ymax, yres) * 1j

        self._zy, self._zx = np.meshgrid(self._x, self._y)
        self._z = self._zx + self._zy

        self._ydim, self._xdim = self._z.shape
        self._z = self._z.flatten()

        self._func = func
        self._f = np.poly1d(func)
        self._fp = np.polyder(self._f)

        self._c = np.zeros(shape = self._z.shape)
        self._uc = np.ones(shape = self._z.shape, dtype=bool)
        self._n = np.arange(len(self._z))

        self._fig = plt.figure(figsize = (16.0, 9.0), dpi = 500)
        ax = plt.subplot(111, xlabel='$x$', ylabel='$iy$', title= str(self._func))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(4)

        self._img = []

    def frame(self):
        for i in it.count():
            fg = self._f(self._z[self._uc])
            nuc = np.abs(fg) > 0.00001
            self._c[self._n[self._uc][~nuc]] = i
            self._uc[self._uc] = nuc
            self._z[self._uc] -= fg[nuc] / self._fp(self._z[self._uc])
            pic = self._c.reshape((self._ydim, self._xdim))
            self._img.append([plt.imshow(pic)])
            if not np.any(nuc):
                for n in range(20):
                    self._img.append([plt.imshow(pic)])
                break

    def animate(self):
        self.frame()
        anim = an.ArtistAnimation(self._fig, self._img, interval  = .001)
        #anim.save('-'.join(it.imap(str, self._func)) + '.avi', writer = 'ffmpeg', fps = 10, dpi = 500)
        plt.show()

f = newton(f, -d, d, -d, d, n, n)
f.animate()