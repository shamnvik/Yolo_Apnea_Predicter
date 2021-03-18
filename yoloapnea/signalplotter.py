import matplotlib.pyplot as plt
import numpy as np
import cv2


class SignalPlotter:

    def __init__(self,image_duration,sliding_window_overlap=None):
        self.image_duration = image_duration
        self.sliding_window_overlap = sliding_window_overlap

    def plot_signal(self,signal):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(signal)
        ax.set_ylim(-1, 1)
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.grid(False)
        plt.axis('off')

        remaining_signal = len(signal)
        start_index = 0

        if remaining_signal < self.image_duration:
            raise Exception("Signal length is too short")

        if self.sliding_window_overlap is None:
            raise Exception("Add sliding window overlap if plotting multiple images")

        while remaining_signal > 0:

            if remaining_signal <= self.image_duration:
                if remaining_signal < self.image_duration:
                    start_index = len(signal) - self.image_duration


                yield start_index,self.plot_part(fig,ax,len(signal) - remaining_signal,len(signal))
                remaining_signal= 0

            elif remaining_signal > self.image_duration:
                yield start_index,self.plot_part(fig,ax,len(signal) - remaining_signal,len(signal) - remaining_signal + self.image_duration)

                #yield start_index,self.signal_to_image(signal[start_index:start_index + self.image_duration])
                start_index = start_index + self.sliding_window_overlap
                remaining_signal -= self.sliding_window_overlap

            else:
                raise Exception("Unhandled else branch")

    def plot_part(self,fig,ax,start,end):
        assert(end,start+self.image_duration)
        ax.set_xlim(start, end)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        plt.close(fig)
        return img



    def signal_to_image(self,signal,start=0,end=None):
        if end is None:
            end = self.image_duration
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(signal)
        ax.set_ylim(-1, 1)

        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.grid(False)
        plt.axis('off')
        ax.set_xlim(start, end)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return img