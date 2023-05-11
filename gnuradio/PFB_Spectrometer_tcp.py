#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Spectrometer
# Author: LUCIANO BAROSI
# Copyright: MIT
# GNU Radio version: 3.10.2.0

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import numpy as np



from gnuradio import qtgui

class PFB_Spectrometer_tcp(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Spectrometer", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Spectrometer")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "PFB_Spectrometer_tcp")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.vec_length = vec_length = 4096
        self.sinc_sample_locations = sinc_sample_locations = np.arange(-np.pi*4/2.0, np.pi*4/2.0, np.pi/vec_length)
        self.sinc = sinc = np.sinc(sinc_sample_locations/np.pi)
        self.samp_rate = samp_rate = 2.048e6
        self.rtl_string = rtl_string = "rtl=0"
        self.prefix = prefix = "../data/raw/GNURADIO/radioCBN"
        self.name = name = "BINGO"
        self.n_samples = n_samples = 100
        self.n_integration = n_integration = 1000
        self.mode = mode = "01"
        self.gain = gain = 50
        self.freq = freq = 1080e6
        self.fit = fit = True
        self.custom_window = custom_window = sinc*np.hamming(4*vec_length)
        self.csv = csv = False

        ##################################################
        # Blocks
        ##################################################
        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            vec_length,
            freq - samp_rate/2,
            samp_rate/vec_length,
            "x-Axis",
            "y-Axis",
            "",
            1, # Number of inputs
            None # parent
        )
        self.qtgui_vector_sink_f_0.set_update_time(0.10)
        self.qtgui_vector_sink_f_0.set_y_axis(0, 10)
        self.qtgui_vector_sink_f_0.enable_autoscale(False)
        self.qtgui_vector_sink_f_0.enable_grid(True)
        self.qtgui_vector_sink_f_0.set_x_axis_units("")
        self.qtgui_vector_sink_f_0.set_y_axis_units("")
        self.qtgui_vector_sink_f_0.set_ref_level(0)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_vector_sink_f_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_vector_sink_f_0.set_line_label(i, labels[i])
            self.qtgui_vector_sink_f_0.set_line_width(i, widths[i])
            self.qtgui_vector_sink_f_0.set_line_color(i, colors[i])
            self.qtgui_vector_sink_f_0.set_line_alpha(i, alphas[i])

        self._qtgui_vector_sink_f_0_win = sip.wrapinstance(self.qtgui_vector_sink_f_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_vector_sink_f_0_win)
        self.qtgui_sink_x_0 = qtgui.sink_c(
            vec_length, #fftsize
            window.WIN_BLACKMAN_hARRIS, #wintype
            freq, #fc
            samp_rate, #bw
            "", #name
            True, #plotfreq
            True, #plotwaterfall
            False, #plottime
            False, #plotconst
            None # parent
        )
        self.qtgui_sink_x_0.set_update_time(1.0/10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.qwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(True)

        self.top_layout.addWidget(self._qtgui_sink_x_0_win)
        self.fft_vxx_0 = fft.fft_vcc(vec_length, True, window.blackmanharris(vec_length), True, 1)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, vec_length,True)
        self.blocks_stream_to_vector_0_0_0_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vec_length)
        self.blocks_stream_to_vector_0_0_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vec_length)
        self.blocks_stream_to_vector_0_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vec_length)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vec_length)
        self.blocks_multiply_const_vxx_0_0_0_0 = blocks.multiply_const_vcc(custom_window[0:vec_length])
        self.blocks_multiply_const_vxx_0_0_0 = blocks.multiply_const_vcc(custom_window[1*vec_length:2*vec_length])
        self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_vcc(custom_window[2*vec_length:3*vec_length])
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_vcc(custom_window[-vec_length:])
        self.blocks_integrate_xx_0_0_0 = blocks.integrate_ff(n_integration, vec_length)
        self.blocks_delay_0_0_0 = blocks.delay(gr.sizeof_gr_complex*1, 3*vec_length)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex*1, 2*vec_length)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, vec_length)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(vec_length)
        self.blocks_add_xx_1 = blocks.add_vcc(1)
        self.blocks_add_xx_0 = blocks.add_vcc(vec_length)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, 1081e6, 20, 0, 0)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_1, 1))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_add_xx_1, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.blocks_add_xx_1, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_integrate_xx_0_0_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.blocks_stream_to_vector_0_0, 0))
        self.connect((self.blocks_delay_0_0, 0), (self.blocks_stream_to_vector_0_0_0, 0))
        self.connect((self.blocks_delay_0_0_0, 0), (self.blocks_stream_to_vector_0_0_0_0, 0))
        self.connect((self.blocks_integrate_xx_0_0_0, 0), (self.qtgui_vector_sink_f_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0_0_0, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.blocks_multiply_const_vxx_0_0_0_0, 0), (self.blocks_add_xx_0, 3))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_stream_to_vector_0_0, 0), (self.blocks_multiply_const_vxx_0_0, 0))
        self.connect((self.blocks_stream_to_vector_0_0_0, 0), (self.blocks_multiply_const_vxx_0_0_0, 0))
        self.connect((self.blocks_stream_to_vector_0_0_0_0, 0), (self.blocks_multiply_const_vxx_0_0_0_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_delay_0_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_delay_0_0_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.qtgui_sink_x_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_squared_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "PFB_Spectrometer_tcp")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_vec_length(self):
        return self.vec_length

    def set_vec_length(self, vec_length):
        self.vec_length = vec_length
        self.set_custom_window(self.sinc*np.hamming(4*self.vec_length))
        self.set_sinc_sample_locations(np.arange(-np.pi*4/2.0, np.pi*4/2.0, np.pi/self.vec_length))
        self.blocks_delay_0.set_dly(self.vec_length)
        self.blocks_delay_0_0.set_dly(2*self.vec_length)
        self.blocks_delay_0_0_0.set_dly(3*self.vec_length)
        self.blocks_multiply_const_vxx_0.set_k(self.custom_window[-self.vec_length:])
        self.blocks_multiply_const_vxx_0_0.set_k(self.custom_window[2*self.vec_length:3*self.vec_length])
        self.blocks_multiply_const_vxx_0_0_0.set_k(self.custom_window[1*self.vec_length:2*self.vec_length])
        self.blocks_multiply_const_vxx_0_0_0_0.set_k(self.custom_window[0:self.vec_length])
        self.blocks_throttle_0.set_sample_rate(self.vec_length)
        self.qtgui_vector_sink_f_0.set_x_axis(self.freq - self.samp_rate/2, self.samp_rate/self.vec_length)

    def get_sinc_sample_locations(self):
        return self.sinc_sample_locations

    def set_sinc_sample_locations(self, sinc_sample_locations):
        self.sinc_sample_locations = sinc_sample_locations
        self.set_sinc(np.sinc(self.sinc_sample_locations/np.pi))

    def get_sinc(self):
        return self.sinc

    def set_sinc(self, sinc):
        self.sinc = sinc
        self.set_custom_window(self.sinc*np.hamming(4*self.vec_length))
        self.set_sinc(np.sinc(self.sinc_sample_locations/np.pi))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.qtgui_sink_x_0.set_frequency_range(self.freq, self.samp_rate)
        self.qtgui_vector_sink_f_0.set_x_axis(self.freq - self.samp_rate/2, self.samp_rate/self.vec_length)

    def get_rtl_string(self):
        return self.rtl_string

    def set_rtl_string(self, rtl_string):
        self.rtl_string = rtl_string

    def get_prefix(self):
        return self.prefix

    def set_prefix(self, prefix):
        self.prefix = prefix

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_n_samples(self):
        return self.n_samples

    def set_n_samples(self, n_samples):
        self.n_samples = n_samples

    def get_n_integration(self):
        return self.n_integration

    def set_n_integration(self, n_integration):
        self.n_integration = n_integration

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        self.mode = mode

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.qtgui_sink_x_0.set_frequency_range(self.freq, self.samp_rate)
        self.qtgui_vector_sink_f_0.set_x_axis(self.freq - self.samp_rate/2, self.samp_rate/self.vec_length)

    def get_fit(self):
        return self.fit

    def set_fit(self, fit):
        self.fit = fit

    def get_custom_window(self):
        return self.custom_window

    def set_custom_window(self, custom_window):
        self.custom_window = custom_window
        self.blocks_multiply_const_vxx_0.set_k(self.custom_window[-self.vec_length:])
        self.blocks_multiply_const_vxx_0_0.set_k(self.custom_window[2*self.vec_length:3*self.vec_length])
        self.blocks_multiply_const_vxx_0_0_0.set_k(self.custom_window[1*self.vec_length:2*self.vec_length])
        self.blocks_multiply_const_vxx_0_0_0_0.set_k(self.custom_window[0:self.vec_length])

    def get_csv(self):
        return self.csv

    def set_csv(self, csv):
        self.csv = csv




def main(top_block_cls=PFB_Spectrometer_tcp, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
