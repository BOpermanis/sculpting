import numpy as np
import cv2
import matplotlib.pyplot as plt
from experiments.hand_segmentation import circle_iterator
from pprint import pprint


def check1():
    h, w = 10, 10
    r_max = 4
    img = np.zeros((h, w), np.uint8)

    pairs = []
    for r, (ih, iw) in circle_iterator(1, 1, h, w, r_max):
        pairs.append((ih, iw))
        print(r, (ih, iw))
    print(len(pairs), len(set(pairs)))

    exit()
    img = cv2.resize(img, (100, 100))

    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    check1()