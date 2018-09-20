from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Highlight NaN/Inf in exr files.')

parser.add_argument(
    'exr_filename',
    help='The exr in which the NaN/Inf pixels need to be found.')

parser.add_argument(
    '--output', type=str,
    help='The png where the pixels are highlighted.')

parsed_arguments, unparsed = parser.parse_known_args()

if not isinstance(parsed_arguments.output, str):
  png_filename, _ = os.path.splitext(parsed_arguments.exr_filename)
  png_filename = png_filename + '.png'
else:
  png_filename = parsed_arguments.output


image_type = cv2.IMREAD_UNCHANGED
image = cv2.imread(parsed_arguments.exr_filename, image_type)

# REMARK: This dummy call avoids an error message (Assertion Failed)
shape = image.shape

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if image.dtype != 'float32':
  image = image.astype(np.float32)


image = np.logical_not(np.isfinite(image))
image = image.astype(np.float32)
image = 255. * image

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite(png_filename, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
