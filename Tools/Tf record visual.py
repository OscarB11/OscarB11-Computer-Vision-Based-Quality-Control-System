import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def parse_tfrecord(tfrecord):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
    }
    return tf.io.parse_single_example(tfrecord, features)

def load_image_from_example(parsed_example):
    image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)
    return image

def visualize_images(tfrecords_paths, num_images=9):
    dataset = tf.data.TFRecordDataset(tfrecords_paths)
    parsed_dataset = dataset.map(parse_tfrecord).shuffle(1000).take(num_images)

    plt.figure(figsize=(15, 15))  # Adjust the size as per your requirement
    for i, parsed_example in enumerate(parsed_dataset):
        image = load_image_from_example(parsed_example)
        plt.subplot(3, 3, i + 1)  # Changed for a 3x3 grid
        plt.imshow(image.numpy())
        xmin = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin']).numpy()
        xmax = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax']).numpy()
        ymin = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin']).numpy()
        ymax = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax']).numpy()
        labels = tf.sparse.to_dense(parsed_example['image/object/class/text'], default_value='').numpy().astype(str)
        for j in range(len(xmin)):
            x1, y1 = xmin[j] * image.shape[1], ymin[j] * image.shape[0]
            x2, y2 = xmax[j] * image.shape[1], ymax[j] * image.shape[0]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.gca().text(x1, y1 - 2, labels[j], bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

tfrecord_files = [
    r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\annotations\test.record',
    r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\annotations\train.record',
    r'boo12\TFODC\Tensorflow\workspace\annotations\TrainMerged.tfrecord',
]

visualize_images(tfrecord_files)
