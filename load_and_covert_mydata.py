import tensorflow as tf
import sys
import utils
from PIL import Image
import numpy as np
import os
import cfg

class DataWriter:
    def Run(self):
        images = self.GetImages()
        labels = self.GetLabels()
        shape = [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNEL]
        with tf.python_io.TFRecordWriter(cfg.PATH_RECORD) as writer:
            self.AddToRecord(images, labels, shape, writer)
        
    def GetImages(self):
        _LoadImg = lambda path: np.array(Image.open(path))
        imgs = []
        for i in range(cfg.NUM_EXAMPLES):
            name = '0' + str(i) + '.jpg'
            full_name = os.path.join(cfg.PATH_IMAGES, name)
            imgs.append(_LoadImg(full_name))
        return np.stack(imgs, axis=0)
        
    def GetLabels(self):
        return list(range(cfg.NUM_EXAMPLES))
        
    def AddToRecord(self, images, labels, shape, tfrecord_writer):
        '''
        Args:
            images: A numpy array of shape [number_of_images, height, width, channels].
            labels: A numpy array of shape [number_of_labels]
            shape: [H, W, CHANNEL)]
            tfrecord_writer: The TFRecord writer to use for writing.
        '''
        W, H, CHANNEL = shape
        num_images = len(labels)

        with tf.Graph().as_default():
            image = tf.placeholder(dtype=tf.uint8, shape=shape)
            encoded_png = tf.image.encode_png(image)
    
            with tf.Session() as sess:
                for j in range(num_images):
                    sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
                    sys.stdout.flush()
    
                    string = sess.run(encoded_png, feed_dict={image: images[j]})
                    img_format = cfg.FORMAT.encode()
                    example = utils.image_to_tfexample(string, img_format, W, H, labels[j], CHANNEL)
                    tfrecord_writer.write(example.SerializeToString())

if __name__=='__main__':
    dw = DataWriter()
    dw.Run()               