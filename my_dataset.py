import tensorflow as tf
slim = tf.contrib.slim
import utils
from tensorflow.contrib.slim.python.slim import queues
import matplotlib.pyplot as plt
import cfg 

def _ShowImg(img):
    plt.figure()
    plt.imshow(img.astype('uint8'))
    plt.show()
    plt.close()
    
def _ShowBatch(imgs):
    num = imgs.shape[0]
    for i in range(num):
        _ShowImg(imgs[i])
    
def _GetDataset():
    _ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [W x H x Channel] image',
        'label': 'A single integer'
    }

    keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                                                    
    }

    shape = [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNEL]
    items_to_handlers = {
            'image': slim.tfexample_decoder.Image(image_key = 'image/encoded',format_key = 'image/format', shape=shape, channels=cfg.IMAGE_CHANNEL),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
            data_sources=cfg.PATH_RECORD,
            reader=tf.TFRecordReader,
            decoder=decoder,
            num_samples=cfg.NUM_EXAMPLES,
            num_classes=cfg.NUM_CLASSES,
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
            labels_to_names=utils.read_label_file(cfg.PATH_LABELS)
            )
    

def _OneEpochTraining(session, images, labels, show=False):
    '''
    this is a small demo,
    but for real project, we need to implement a training process for one epoch here
    '''
    
    try:
        while True:
            val_images, val_labels = session.run([images, labels])
            print('label is {}'.format(val_labels))
            if show and len(images.shape)==4:
                _ShowBatch(val_images)
            if show and len(images.shape)==3:
                _ShowImg(val_images)
    except tf.errors.OutOfRangeError:
        print('Done training -- this is the end of epoch')
    return
        
        
def HowToRunOneEpoch():
    '''
    demonstrate how to run one epoch with shuffling
    '''
    my_dataset = _GetDataset()
    provider = slim.dataset_data_provider.DatasetDataProvider(
                    my_dataset, 
                    num_readers=4,  # The number of parallel readers that read data from the dataset
                    shuffle=True,   # Whether to shuffle the data sources and common queue when reading
                    num_epochs=1,   # The number of times each data source is read. If left as None, the data will be cycled through indefinitely.
                    common_queue_capacity = 20*cfg.BATCH_SIZE,   # The capacity of the common queue.
                    common_queue_min= 10*cfg.BATCH_SIZE,         # The minimum number of elements in the common queue after a dequeue.
                    ) 
    
    [image, label] = provider.get(['image', 'label'])
    
    '''
    preprocessing
    here is a demo with one hot transformation
    in real project, we need to perform data augmentation here
    '''
    label = slim.one_hot_encoding(label, cfg.NUM_CLASSES)
    
    # note, num_epochs must be initialized by local variables initializer
    op_init = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(op_init)
        with queues.QueueRunners(sess):
            _OneEpochTraining(sess, image, label, show=True)
                    
                
      

if __name__=='__main__':
    HowToRunOneEpoch()
    #HowToRunOneEpochWithBatch()