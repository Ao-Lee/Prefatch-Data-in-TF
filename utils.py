import tensorflow as tf

def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
        values: A scalar or list of values.

    Returns:
        A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
        values: A string.

    Returns:
        A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    """Returns a TF-Feature of floats.

    Args:
        values: A scalar of list of values.

    Returns:
        A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))
    
def image_to_tfexample(image_data, image_format, height, width, class_id, channel):
    return tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(image_data),
            'image/format': bytes_feature(image_format),
            'image/class/label': int64_feature(class_id),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/channel': int64_feature(channel)
    }))
    
def read_label_file(filename):
    """Reads the labels file and returns a mapping from ID to class name.
    
    Returns:
        A map from a label (integer) to class name.
    """

    with tf.gfile.Open(filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names
