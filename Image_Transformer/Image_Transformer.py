import numpy as np
from PIL import Image, ImageFilter

from .diff_pix import diff_pixelate
import unittest
import os

class Image_Transformer:

	def __init__(self, method):
		self.method = method
		self.blur_types = ['BOX', 'GAUSSIAN', 'DP']

	def pil2numpy(self, frame):
		result = np.asarray(frame, dtype=np.float32)
		return result

	def numpy2pil(self, frame):
		result = Image.fromarray(frame.astype(np.uint8))
		return result

	def apply_transformation(self, frame, m):
		if self.method == self.blur_types[0]:
			return frame.filter(ImageFilter.BoxBlur(0))
		elif self.method == self.blur_types[1]:
			return frame.filter(ImageFilter.GaussianBlur(52))
		elif self.method == self.blur_types[2]:
			w, h = frame.size
			numframe = self.pil2numpy(frame)

			pix = diff_pixelate(numframe, h, w, m, 10, 8)

			result = self.numpy2pil(pix)

			return result
		else:
			print('Incorrect Blur Method name')
			print('Available blurring methods are:')

			for t in self.blur_types:
				print(t)

			return None

class Test_Image_Transformer(unittest.TestCase):
	def test_box_blur(self):
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		image_transformer = Image_Transformer('BOX')

		img2 = image_transformer.apply_transformation(img, 10)
		img2.save(os.path.join(directory, 'test_images/sample_image_box_blur.png'))

	def test_gaussian_blur(self):
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		image_transformer = Image_Transformer('GAUSSIAN')

		img2 = image_transformer.apply_transformation(img, 10)
		img2.save(os.path.join(directory, 'test_images/sample_image_gaussian_blur.png'))

	def test_dp_blur(self):
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		image_transformer = Image_Transformer('DP')

		img2 = image_transformer.apply_transformation(img, 10)
		img2.save(os.path.join(directory, 'test_images/sample_image_dp_blur.png'))

if __name__=='__main__':
	unittest.main()