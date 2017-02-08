# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque

def curvature(fit_model, ploty):
  """
  Caculate lane curvature
  -----------------------
  Function to calculate the radius of curvature based on pixel values,
  so the radius we are reporting is in pixel space,
  which is not the same as real world space.
  """
  y_eval = np.max(ploty) / 2.
  radius = ((1 + (2 * fit_model[0] * y_eval + fit_model[1])**2)**1.5) / np.absolute(2 * fit_model[0])
  return radius

def distance_to_center(x):
  """
  Calculate distance from x to center
  -------------------------
  The offset of the lane center from the center of the image
  (converted from pixels to meters) is the distance from
  the center of the lane.
  """
  img_center = 1280. / 2
  xm_per_pix = 3.7 / 700 # meters per pixel in x dimension
  position = (img_center - x) * xm_per_pix
  return position

class Line():
  """
  Class to receive the characteristics of each line detection.
  """
  def __init__(self):
    # was the line detected in the last iteration?
    self.detected = False
    # x values of the last n fits of the line
    self.recent_xfitted = deque(maxlen=5)
    # average x values of the fitted line over the last n iterations
    self.bestx = None
    # polynomial coefficients averaged over the last n iterations
    self.best_fit = None
    # polynomial coefficients for the most recent fit
    self.current_fit = None
    # radius of curvature of the line in some units
    self.radius_of_curvature = None
    # distance in meters of vehicle center from the line
    self.line_base_pos = None
    # difference in fit coefficients between last and new fits
    self.diffs_fit = np.array([0,0,0], dtype='float')
    self.diff_curvature = None
    # x values for detected line pixels
    self.allx = None
    # y values for detected line pixels
    self.ally = None

  def set_lane(self, x, y, fit):
    """
    Function to define the lane, given the x and y points, and the
    second order polynomial. With this data, the position and curvature
    values are set. Finally, it calculates a mean of the last 5 polynomials

    """
    self.allx = x
    self.ally = y
    self.line_base_pos = distance_to_center(self.allx[0])

    if self.current_fit == None:
      self.current_fit = fit
      self.radius_of_curvature = curvature(fit, self.ally)

    # Diff with previous values
    self.diffs_fit = self.current_fit - fit
    self.diff_curvature = np.absolute(self.radius_of_curvature - curvature(fit, self.ally))

    # Update
    self.recent_xfitted.append(fit)
    self.bestx = sum(self.recent_xfitted) / len(self.recent_xfitted)
    self.current_fit = fit
    self.radius_of_curvature = curvature(self.bestx, self.ally)


  def get_x(self):
    return self.allx

  def get_y(self):
    return self.ally

  def get_fit(self):
    # return self.current_fit
    return self.bestx

  def get_curvature(self):
    return self.radius_of_curvature

  def get_dist_to_center(self):
    return self.line_base_pos


class Lanes():
  """
  Class to find lanes given a warped binary image
  """

  def __init__(self):
    self.frame_number = 0  # Frame counter (used for finding new lanes)
    self.left = Line()
    self.right = Line()
    self.old_polygon = None
    self.polygon_diff = 0

  def find_lane_points(self, binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, deg=2)
    right_fit = np.polyfit(righty, rightx, deg=2)

    return leftx, lefty, left_fit, rightx, righty, right_fit

  def fast_find_lane_points(self, binary_warped):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = (
      (nonzerox > (self.left.get_fit()[0]*(nonzeroy**2) + self.left.get_fit()[1]*nonzeroy + self.left.get_fit()[2] - margin)) &
      (nonzerox < (self.left.get_fit()[0]*(nonzeroy**2) + self.left.get_fit()[1]*nonzeroy + self.left.get_fit()[2] + margin)))
    right_lane_inds = (
      (nonzerox > (self.right.get_fit()[0]*(nonzeroy**2) + self.right.get_fit()[1]*nonzeroy + self.right.get_fit()[2] - margin)) &
      (nonzerox < (self.right.get_fit()[0]*(nonzeroy**2) + self.right.get_fit()[1]*nonzeroy + self.right.get_fit()[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return leftx, lefty, left_fit, rightx, righty, right_fit

  def cast_lane(self, img, Minv):
    """
    Cast the fitted line back to the original image
    """
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = self.left.get_fit()[0]*ploty**2 + self.left.get_fit()[1]*ploty + self.left.get_fit()[2]
    right_fitx = self.right.get_fit()[0]*ploty**2 + self.right.get_fit()[1]*ploty + self.right.get_fit()[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    lane = np.int_([pts])
    cv2.fillPoly(color_warp, lane, (0, 255, 0))

    # cv2.matchShapes, compares two shapes and returns a similarly index,
    # with 0 being identical shapes. We use this to make sure the
    # polygon for the next frame is close to what it is expected to look
    # like and if not we can elect to use old polygon instead.
    new_polygon = lane[0]
    if self.old_polygon != None:
      self.polygon_diff = cv2.matchShapes(self.old_polygon, new_polygon, 1, 0.0)
      if self.polygon_diff < 0.045:
        # Use the new polygon points to write the next frame due to
        # similarites of last sucessfully written polygon area
        self.old_polygon = new_polygon
    else:
      self.old_polygon = new_polygon


    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    img_combined = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return img_combined


  def find_and_draw(self, img, binary_warped, Minv):

    leftx, lefty, left_fit, rightx, righty, right_fit = self.find_lane_points(binary_warped)

    self.left.set_lane(leftx, lefty, left_fit)
    self.right.set_lane(rightx, righty, right_fit)

    img_lane = self.cast_lane(img, Minv)

    # Text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Left lane to center: {:.2f}m".format(self.left.get_dist_to_center())
    cv2.putText(img_lane, text, (250, 120), font, 1.2, (255, 255, 255), 2)

    avg_curv = (self.left.get_curvature() + self.right.get_curvature()) / 2
    text = "Lane curvature: {} m".format(int(avg_curv))
    cv2.putText(img_lane, text, (250, 160), font, 1.2, (255, 255, 255), 2)

    # Update frame counter
    self.frame_number += 1

    return img_lane
