
import numpy as np

class FIFO:
    def __init__(self, history=3, bincount=5, collecting=False,
                 collect_count=10000, smoothing=True, smooth_val=0.8, _dtype=int, **kwargs):

        history = kwargs.get('history', history)
        bincount = kwargs.get('bincount', bincount)
        smooth_val = kwargs.get('smooth_val', smooth_val)
        smoothing = kwargs.get('smoothing', smoothing)
        collecting = kwargs.get('collecting', collecting)
        collect_count = kwargs.get('collect', collect_count)
        _dtype = kwargs.get('_dtype', _dtype)


        self.collecting = collecting
        self.collect_bin = np.zeros([collect_count, bincount])
        self.collect_count = collect_count
        self.smoothing = smoothing
        self.index = 0
        self.h = history
        self.bs = bincount
        self.all_bins = np.zeros([history, bincount], dtype=_dtype)
        self.smootharray = np.array([smooth_val**i for i in range(self.h)])

    def put(self, new_bins):
        self.all_bins[self.index % self.h] = new_bins
        #print(self.all_bins)
        if self.smoothing:
            self.smootharray = np.roll(self.smootharray, 1)
        self.index += 1
        if self.collecting:
            self.collect_bin[self.index] = new_bins
            if self.index == self.collect_count:
                np.save(self.songname, self.collect_bin)

    def getsum(self):
        return np.sum(self.all_bins, axis=0)

    def getsmoothed(self):
        return np.dot(self.smootharray, self.all_bins)

    def setTranslateParams(self, old_low, old_high, new_low, new_high):
        old = np.array([[old_low, 1], [old_high, 1]])
        new = np.array([new_low, new_high])
        self.TP = np.linalg.solve(old, new)


if __name__ == '__main__':
    fifo = FIFO()
    for i in range(20):
        fifo.put([x for x in np.arange(i, 5+i)])
        print(fifo.getsum())
        print(fifo.all_bins)
