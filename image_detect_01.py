import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

def matchAB(fileA, fileB):
    # 读取图像数据
    imgA = cv2.imread(fileA)
    imgB = cv2.imread(fileB)

    # 转换成灰色
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # akaze特征量抽出
    akaze = cv2.AKAZE_create()
    kpA, desA = akaze.detectAndCompute(grayA, None)
    kpB, desB = akaze.detectAndCompute(grayB, None)

    # BFMatcher定义和图形化
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desB, desB)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_image = cv2.drawMatches(imgA, kpA, imgB, kpB, matches, None, flags=2)

    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_image',
        type=str,
        default='img/image01-0.png',
        help='source image'
    )

    parser.add_argument(
        '--target_image',
        type=str,
        default='img/image01-1.png',
        help='target image'
    )

    FLAGS, unparsed = parser.parse_known_args()

    matchAB(FLAGS.source_image, FLAGS.target_image)