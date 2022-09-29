import os
import cv2
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from icrawler.builtin import GoogleImageCrawler


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'ResizeBilinear_3:0'
    INPUT_SIZE = 513
    label = 15

    def __init__(self, path):
        self.graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef.FromString(open(path, 'rb').read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def get_mask(self, image):
        height, width = image.shape[:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = softmax(batch_seg_map[0][:target_size[1], :target_size[0]], axis=-1)
        return seg_map[:, :, self.label]

    def transform(self, image, mask, query):
        mask = cv2.resize(mask, image.shape[:2][::-1])[:, :, np.newaxis]
        x0, y0, c0 = image.shape
        if query != 'bokeh':
            try:
                os.mkdir(query)
            except:
                pass
            google_crawler = GoogleImageCrawler(storage={'root_dir': f'/tmp/{query}'})
            google_crawler.crawl(keyword=query, max_num=1)
            background = cv2.imread(f'/tmp/{query}/000001.jpg')
            x, y, c = background.shape
            new_x = x * y0 / y
            new_y = y * x0 / x
            if new_x > x0:
                new_y = y0
            else:
                new_x = x0
            background = cv2.resize(background, (int(new_y), int(new_x)))[:x0, :y0]
        else:
            background = cv2.blur(image.copy(), (x0 // 10, y0 // 10))
            print(background.shape)
        print('blend', image.shape, mask.shape, background.shape)
        new_img = image * mask + background * (1 - mask)
        return new_img
