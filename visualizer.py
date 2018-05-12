"""Module that contains visualizer classes."""

import sys
import random
import time


import numpy as np
from PySide import QtCore, QtGui
from math import floor, sqrt, pi, exp, log

from FIFO import *


SAMPLE_MAX = 32767
SAMPLE_MIN = -(SAMPLE_MAX + 1)
SAMPLE_RATE = 44100 # [Hz]
NYQUIST = SAMPLE_RATE / 2
SAMPLE_SIZE = 16 # [bit]
CHANNEL_COUNT = 1
BUFFER_SIZE = 2000


def gaussian(x_array, sigma, slen, Print=False):
    mu = np.sum(x_array) / slen
    exppart = np.exp(-1./2. * ((x_array - mu)/sigma)**2)
    mulitplier = 1. / (sigma * sqrt(2. * pi)) * exppart
    if Print:
        print "mu =", mu
        print "exppart =", exppart
        print "mulitplier =", mulitplier
    return mulitplier

def gausssum(_array, Print=False):
    slen = np.size(_array)
    sigma = slen/2.53333
    if Print:
        print "slen", slen
        print "_array", _array
        print "gaussian", gaussian(_array, sigma, slen, Print=Print)
    return np.sum(_array*gaussian(_array, sigma, slen, Print=False))

def gaussumm_array(l_o_l):
    return [gausssum(l_o_l[i]+l_o_l[i+1]+l_o_l[i+2], Print=False) for i in range(len(l_o_l)-2)]



def intensify(arraylike, intensity):
    exparr = np.array([intensity**i for i in range(len(arraylike))])
    return np.multiply(exparr.T, arraylike)


class Visualizer(QtGui.QLabel):
    """The base class for visualizers.

    When initializing a visualizer, you must provide a get_data function which
    takes no arguments and returns a NumPy array of PCM samples that will be
    called exactly once each time a frame is drawn.

    Note: Although this is an abstract class, it cannot have a metaclass of
    abcmeta since it is a child of QObject.
    """
    def __init__(self, update_interval=25, **kwargs):
        super(Visualizer, self).__init__()
        update_interval = kwargs.get("update_interval", update_interval)

        self.get_data = kwargs["get_data"]
        self.update_interval = update_interval #33ms ~= 30 fps
        self.sizeHint = lambda: QtCore.QSize(400, 400)
        self.setStyleSheet('background-color: black;');
        title = self.createTitle(update_interval, **kwargs)
        self.setWindowTitle(title)

    def createTitle(self, update_interval, **kwargs):
        return 'PyVisualizer ' + ' '.join(["update_interval" + str(update_interval)] + [key + str(value) for key, value in kwargs.iteritems()])



    def show(self):
        """Show the label and begin updating the visualization."""
        super(Visualizer, self).show()
        self.refresh()


    def refresh(self):
        """Generate a frame, display it, and set queue the next frame"""
        data = self.get_data()
        interval = self.update_interval
        if data is not None:
            t1 = time.clock()
            #print data
            self.setPixmap(QtGui.QPixmap.fromImage(self.generate(data)))
            #decrease the time till next frame by the processing time so that the framerate stays consistent
            interval -= 1000 * (time.clock() - t1)
        if self.isVisible():
            QtCore.QTimer.singleShot(self.update_interval, self.refresh)

    def generate(self, data):
        """This is the abstract function that child classes will override to
        draw a frame of the visualization.

        The function takes an array of data and returns a QImage to display"""
        raise NotImplementedError()


class LineVisualizer(Visualizer):
    """This visualizer will display equally sized rectangles
    alternating between black and another color, with the height of the
    rectangles determined by frequency, and the quantity of colored rectanges
    influnced by amplitude.
    """

    def __init__(self, get_data, columns=2):
        super(LineVisualizer, self).__init__(get_data=get_data)

        self.columns = columns
        self.brushes = [QtGui.QBrush(QtGui.QColor(255, 255, 255)), #white
                        QtGui.QBrush(QtGui.QColor(255, 0, 0)),     #red
                        QtGui.QBrush(QtGui.QColor(0, 240, 0)),     #green
                        QtGui.QBrush(QtGui.QColor(0, 0, 255)),     #blue
                        QtGui.QBrush(QtGui.QColor(255, 255, 0)),   #yellow
                        QtGui.QBrush(QtGui.QColor(0, 255, 255)),   #teal
                        ]
        self.brush = self.brushes[3]

        self.display_odds = True
        self.display_evens = True
        self.is_fullscreen = False

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_I:
            self.display_evens = True
            self.display_odds = True
        elif event.key() == QtCore.Qt.Key_O:
            self.display_evens = True
            self.display_odds = False
        elif event.key() == QtCore.Qt.Key_P:
            self.display_evens = False
            self.display_odds = True
            return
        elif event.key() == QtCore.Qt.Key_Escape:
            if self.is_fullscreen:
                self.showNormal()
                self.is_fullscreen = False
            else:
                self.showFullScreen()
                self.is_fullscreen = True
        else:
            #Qt.Key enum helpfully defines most keys as their ASCII code,
            #   so we can use ord('Q') instead of Qt.Key.Key_Q
            color_bindings = dict(zip((ord(i) for i in 'QWERTYU'), self.brushes))
            try:
                self.brush = color_bindings[event.key()]
            except KeyError:
                if QtCore.Qt.Key_0 == event.key():
                    self.columns = 10
                elif QtCore.Qt.Key_1 <= event.key() <= QtCore.Qt.Key_9:
                    self.columns = event.key() - QtCore.Qt.Key_1 + 1

    def generate(self, data):
        fft = np.absolute(np.fft.rfft(data, n=len(data)))
        freq = np.fft.fftfreq(len(fft), d=1./SAMPLE_RATE)
        max_freq = abs(freq[fft == np.amax(fft)][0])
        max_amplitude = np.amax(data)

        rect_width = int(self.width() / (self.columns * 2))

        freq_cap = 20000. #this determines the scale of lines
        if max_freq >= freq_cap:
            rect_height = 1
        else:
            rect_height = int(self.height() * max_freq / freq_cap)
            if rect_height == 2: rect_height = 1


        img = QtGui.QImage(self.width(), self.height(), QtGui.QImage.Format_RGB32)
        img.fill(0) #black


        if rect_height >= 1:
            painter = QtGui.QPainter(img)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(self.brush)

            for x in xrange(0, self.width() - rect_width, rect_width * 2):
                for y in xrange(0, self.height(), 2 * rect_height):
                    if random.randint(0, int(max_amplitude / float(SAMPLE_MAX) * 10)):
                        if self.display_evens:
                            painter.drawRect(x, y, rect_width, rect_height)
                        if self.display_odds:
                            painter.drawRect(x + rect_width, self.height() - y - rect_height, rect_width, rect_height)

            del painter #
        print img
        return img

class Spectrogram(Visualizer):
    def __init__(self, **kwargs):
        """

        :param kwargs: argdict
        :param bincount: subdivision in number of bins
        :param history: past timepoints to take into account
        :param hightone_distance: number to cerrect for freqency length
            of high tones. since they travel less far this has to be
            corrected for, depends on speaker/recorder (tweaking)
        :param amplitude_difference: exponential value of difference between low & high values
        """
        #Todo add formula for amplitude difference to heigh correction

        Visualizer.__init__(self, **kwargs)
        bincount = kwargs["bincount"]
        self.fifo = FIFO(**kwargs)  #magic number
        self.h = kwargs["history"]
        self.htd = kwargs["hightone_distance"]
        self.ampt_d = kwargs["amplitude_difference"]
        self.fc = kwargs["fourier_count"]
        self.fd = log(float(self.fc)/1000.)/log(2)
        self.bc = bincount
        self.bd = bincount/120.         # magic number ez but works
        self.bcf = float(bincount)
        self.curbins = np.zeros(bincount)
        self.last = None





    def generate(self, data):
        bc = self.bc
        #print "data", data
        fft = np.absolute(np.fft.rfft(data, self.fc))


        curbins = np.zeros(bc)

        fft_len = len(fft)
        step = 1


        seq4 = [0.] + [2. ** (x / self.bcf ) for x in range(bc)]
        #seq4 = [seq4[i]/(2.*self.fd) for i in range(len(seq4))]
        steps = [int(sum(seq4[:i+1])) for i in range(len(seq4))]
        #print steps
        #print("len fft", len(fft))
        for i in xrange(len(curbins)):
            # curbins[i] = np.sum(fft[i])
            #curbins[i] = np.sum(fft[i+int(seq1[i]):i + step + int(seq1[i+1])])
            curbins[i] = np.mean(fft[steps[i]:steps[i+1]])


        self.fifo.put(curbins)
        totbins = self.fifo.getsmoothed()
        tbs = np.sign(totbins)
        tba = np.absolute(totbins)
        tbas = np.sqrt(tba)
        #print(tbs)
        #print(tbas)
        work_values = np.multiply(tbas, tbs)
        work_values = intensify(work_values, self.htd)

        img = QtGui.QImage(self.width(), self.height(), QtGui.QImage.Format_RGB32)
        img.fill(0)
        painter = QtGui.QPainter(img)
        painter.setPen(QtCore.Qt.NoPen)
        print work_values
        for i, bin in enumerate(np.nditer(work_values)):
            height = (self.height() * (bin + 0.01) / (sqrt(float(SAMPLE_MAX)) * 16))**self.ampt_d / 10
            width = self.width() / (2*self.bcf)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(i/self.bcf*254, 255, 255)))
            painter.drawRect(i * width, self.height()/2 - height, width, height)
            painter.drawRect(i * width, self.height()/2 + height, width, -height)
            painter.drawRect(-i * width + self.width(), self.height() / 2 - height, width, height)
            painter.drawRect(-i * width + self.width(), self.height() / 2 + height, width, -height)

        #print(img)
        del painter

        return img

