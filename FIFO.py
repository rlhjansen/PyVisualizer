
import numpy as np

class FIFO:
    def __init__(self, history=3, bincount=5, smoothing=True, smoothVal=0.8, **kwargs):
        history = kwargs.get('history', history)
        bincount = kwargs.get('bincount', bincount)
        smoothing = kwargs.get('smoothVal', smoothVal)
        smoothing = kwargs.get('smoothing', smoothing)

        self.smoothing = smoothing
        self.index = 0
        self.h = history
        self.bs = bincount
        self.all_bins = np.zeros([history, bincount], dtype=int)
        self.smootharray = np.array([smoothVal**i for i in range(self.h)])

    def put(self, new_bins):
        self.all_bins[self.index % self.h] = new_bins
        print(self.all_bins)
        if self.smoothing:
            self.smootharray = np.roll(self.smootharray, 1)
        self.index += 1

    def getsum(self):
        return np.sum(self.all_bins, axis=0)

    def getsmoothed(self):
        return np.dot(self.smootharray, self.all_bins)

if __name__ == '__main__':
    fifo = FIFO()
    for i in range(20):
        fifo.put([x for x in np.arange(i, 5+i)])
        print(fifo.getsum())
        print(fifo.all_bins)
