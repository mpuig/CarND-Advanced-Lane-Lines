# -*- coding: utf-8 -*-
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from camera import Camera
from lanes import Lanes
from moviepy.editor import VideoFileClip
import glob

DUMP = True

camera = Camera()
lanes = Lanes()

def process_image(img):
    base = "./output_images/frame" + str(lanes.frame_number)

    # Pipeline
    undist = camera.undistort_image(img)
    warped = camera.warp(undist)
    binary_warped = camera.multi_thresholds(warped)
    combined = lanes.find_and_draw(img, binary_warped, Minv=camera.Minv)

    if DUMP:
        mpimg.imsave(base + "_0.jpg", img)
        #mpimg.imsave(base + "_1_undist.jpg", undist)
        #mpimg.imsave(base + "_2_warped.jpg", warped)
        #mpimg.imsave(base + "_3_binary_warped.jpg", binary_warped)
        mpimg.imsave(base + "_4_combined.jpg", combined)

    return combined


def main():
    print("Running main...")
    clip = VideoFileClip("./project_video.mp4")
    output_video = "./project_video_processed4.mp4"
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)

def test():
    print("Running test...")
    filenames = glob.glob("./test_images/test*.jpg")
    for filename in filenames:
        print("processing", filename)
        img = cv2.imread(filename)
        img_lane = process_image(img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Lane Lines Detection')
    parser.add_argument('mode', type=str, help='Execution mode')
    args = parser.parse_args()

    if args.mode == 'test':
        test()
    else:
        main()
