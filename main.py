"""Special thanks to ajalt: https://github.com/ajalt/PyVisualizer

"""


#!/usr/bin/python

import sys

import numpy as np
from PySide import QtCore, QtGui

from visualizer import *

app = QtGui.QApplication(sys.argv)

def record_qt_multimedia():
    info = QtMultimedia.QAudioDeviceInfo.defaultInputDevice()
    print "info", info
    format = info.preferredFormat()
    format.setChannels(CHANNEL_COUNT)
    format.setChannelCount(CHANNEL_COUNT)
    format.setSampleSize(SAMPLE_SIZE)
    format.setSampleRate(SAMPLE_RATE)

    if not info.isFormatSupported(format):
        print 'Format not supported, using nearest available'
        format = nearestFormat(format)
        if format.sampleSize != SAMPLE_SIZE:
            #this is important, since effects assume this sample size.
            raise RuntimeError('16-bit sample size not supported!')

    audio_input = QtMultimedia.QAudioInput(format, app)
    audio_input.setBufferSize(BUFFER_SIZE)
    source = audio_input.start()

    def read_data():
        data = np.fromstring(source.readAll(), 'int16').astype(float)
        if len(data):
            return data
    return read_data

def record_pyaudio():
    p = pyaudio.PyAudio()

    stream = p.open(format = pyaudio.paInt16,
                    channels = CHANNEL_COUNT,
                    rate = SAMPLE_RATE,
                    input = True,
                    frames_per_buffer = BUFFER_SIZE)

    def read_data():
        data = np.fromstring(stream.read(stream.get_read_available()), 'int16').astype(float)
        if len(data):
            return data
    return read_data

try:
    from PySide import QtMultimedia
    print 'Using PySide'
    read_data = record_qt_multimedia()
except ImportError:
    print 'Using PyAudio'
    import pyaudio
    read_data = record_pyaudio()


window = Spectrogram(get_data=read_data, history=8, bincount=30, smooth_val=0.95)
window.show()
app.exec_()
