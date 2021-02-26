from unittest import TestCase
import numpy as np
import cv2

import yoloapnea.signalplotter as signalplotter

image_duration = 900
overlap = 450

class TestSignalPlotter(TestCase):

    def setUp(self):

        test_signal = np.load("shhs1-200753-signal.npz")
        self.abdo_signal = test_signal["abdo_res"]

        self.plotter = signalplotter.SignalPlotter()

    def test_plot_signal_correct_length(self):
        part_signal = self.abdo_signal[0:image_duration]
        images = self.plotter.plot_signal(part_signal)
        for img in images:
            shape = img.shape
            self.assertEqual(shape[0],1000)
            self.assertEqual(shape[1],1000)
            self.assertEqual(shape[2],3)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_plot_signal_short_length(self):
        part_signal = self.abdo_signal[0:image_duration-200]

        with self.assertRaises(Exception):
            self.plotter.plot_signal(part_signal)



    def test_plot_signal_two_images(self):
        part_signal = self.abdo_signal[0:image_duration+overlap]
        images = self.plotter.plot_signal(part_signal)

        image_count = 0

        for img in images:
            shape = img.shape
            self.assertEqual(shape[0], 1000)
            self.assertEqual(shape[1], 1000)
            self.assertEqual(shape[2], 3)
            image_count += 1

        self.assertEqual(2,image_count)

    def test_plot_signal_three_images(self):
        part_signal = self.abdo_signal[0:image_duration + overlap+overlap]
        images = self.plotter.plot_signal(part_signal)

        image_count = 0

        for img in images:
            shape = img.shape
            self.assertEqual(shape[0], 1000)
            self.assertEqual(shape[1], 1000)
            self.assertEqual(shape[2], 3)
            image_count += 1

        self.assertEqual(3, image_count)

    def test_plot_signal_data_for_one_and_a_half_image(self):
        part_signal = self.abdo_signal[0:image_duration + int((overlap/2))]

        images = self.plotter.plot_signal(part_signal)

        image_count = 0

        for img in images:
            shape = img.shape
            self.assertEqual(shape[0], 1000)
            self.assertEqual(shape[1], 1000)
            self.assertEqual(shape[2], 3)
            image_count += 1

        self.assertEqual(2, image_count)


    def test_plot_signal_data_for_two_and_a_half_image(self):
        part_signal = self.abdo_signal[0:image_duration + overlap + int((overlap/2))]

        images = self.plotter.plot_signal(part_signal)

        image_count = 0

        for start_index, img in images:
            shape = img.shape
            self.assertEqual(shape[0], 1000)
            self.assertEqual(shape[1], 1000)
            self.assertEqual(shape[2], 3)
            image_count += 1
            print(start_index)


        self.assertEqual(3, image_count)