import tensorflow as tf

from PIL import Image, ImageEnhance
import os
import numpy as np
import cv2
import scipy.misc
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Dense, Flatten, Dropout


def image_enhancement(file):
	src_file = 'data/files/'+ file
	dest = 'data/contrast'
	im = Image.open(src_file)
	newsize = (256, 256) 
	im1 = im.resize(newsize)

	enhancer = ImageEnhance.Contrast(im1)
	factor = 2 #increase contrast
	im_output = enhancer.enhance(factor)
	file_path = os.path.join(dest,file)
	im_output.save(file_path)

def build_filters():
	filters = []
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi / 16):
		kern = cv2.getGaborKernel((ksize, ksize), 0.7, theta, 15.0, 0.2, 1, ktype=cv2.CV_32F)
		kern /= 1.5*kern.sum()
		filters.append(kern)
	return filters

def process(img, filters):
	accum = np.zeros_like(img)
	for kern in filters:
		fimg = cv2.filter2D(np.array(img), cv2.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum


def rib_suppression(file):
	dest = 'data/rib_suppression'
	src = 'data/contrast/' + file
	im = Image.open(src)
	filters = build_filters()
	res1 = process(im, filters)
	file_path = os.path.join(dest,file)
	img = Image.fromarray(res1)
	if img.mode in ("RGBA", "P"):
		img = img.convert("RGB")
	img.save(file_path)

# def model_output(file):
# 	model = load_model('data/models/vgg_E20_B64_D2.h5', custom_objects={'Dense': Dense, 'Flatten':Flatten, 'Dropout': Dropout})
# 	model.load_weights('data/models/vgg_E20_B64_D2.h5')
# 	model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])
# 	model.summary()
# 	fpath = 'data/rib_suppression/'+ file
# 	image = tf.io.read_file(fpath)
# 	image = tf.image.decode_jpeg(image, channels=3)
# 	print(image.shape)
# 	image = tf.expand_dims(image, axis=0)
# 	with sess.as_default():
# 		with sess.graph.as_default():
# 			y_prob = model.predict(image)
# 	print(y_prob)
# 	y_classes = y_prob.argmax(axis=-1)
# 	print(y_classes)
# 	class_map = {0: 'COVID-19', 1: 'Normal', 2: 'Pneumonia'}
# 	data = class_map[y_classes[0]]
# 	print(data)
# 	return data


def getfile(name):
	image_enhancement(name)
	rib_suppression(name)
	# output = model_output(name)
	# return output