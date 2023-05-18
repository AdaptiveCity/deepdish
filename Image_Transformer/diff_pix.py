import numpy as np
from PIL import Image
import cv2
import unittest
import os

def laplace_noise(frame, mean, std):
	frame = frame/255.0
	dims = frame.shape

	noise = np.random.laplace(mean, std, frame.shape)
	new_frame = np.clip(frame + noise, 0, 1) * 255

	return new_frame

def gaussian_noise(frame, mean, std):
	frame = frame/255.0
	dims = frame.shape

	noise = np.random.normal(mean, std, dims)
	new_frame = np.clip(frame + noise, 0, 1) * 255

	return new_frame

def pixelate_resize(frame, height, width, scale):
	frame = cv2.resize(frame, (int(width/scale), int(height/scale)), interpolation=cv2.INTER_LINEAR)

	frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

	return frame

def pixelate_rgb(img, n):
	width, height = img.shape

	x_end = width - (width % n)
	y_end = height - (height % n)

	result = np.zeros((x_end, y_end, 4))

	for x in range(0, x_end, n):
		for y in range(0, y_end, n):
			window = img[x:x+n,y:y+n]
			mean_val = window.mean(axis=(0,1))

			result[x:x+n,y:y+n] = mean_val

	return result

def diff_pixelate(frame, height, width, m, e, scale):
	temp = False

	if len(frame.shape) == 2:
		frame = np.expand_dims(frame, axis=2)
		temp = True

	new_frame = frame

	new_frame = pixelate_resize(frame, height, width, scale)

	result = np.zeros(new_frame.shape)

	a = (frame.shape[2] * m) / (e * scale * scale)

	for i in range(frame.shape[2]):
		result[:,:,i] = laplace_noise(new_frame[:,:,i], 0, 0.1)

	if temp:
		result = np.squeeze(result)

	return result

class Test_Diff_Pix(unittest.TestCase):
	def test_laplace_noise(self):
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img = np.asarray(img, dtype=np.float32)


		img2 = laplace_noise(img, 0.5, 0.1)
		img2 = Image.fromarray(img2.astype(np.uint8))
		img2.show()

	def test_gaussian_noise(self):
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img = np.asarray(img, dtype=np.float32)

		img2 = gaussian_noise(img, 0.5, 0.1)
		img2 = Image.fromarray(img2.astype(np.uint8))
		img2.show()

	def test_pixelate_image(self):
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		width, height = img.size
		img = np.asarray(img, dtype=np.float32)

		img2 = pixelate_resize(img, height, width, 8)
		img2 = Image.fromarray(img2.astype(np.uint8))
		img2.save(os.path.join(directory, 'test_images/sample_image_pix.png'))

	def test_diff_pix(self):
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		width, height = img.size
		img = np.asarray(img, dtype=np.float32)


		img2 = diff_pixelate(img, height, width, 10, 10, 8)
		img2 = Image.fromarray(img2.astype(np.uint8))
		img2.save(os.path.join(directory, 'test_images/sample_image_diff_pix.png'))

if __name__=='__main__':
	unittest.main()