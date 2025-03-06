import tensorflow as tf

def is_tfrecord_corrupted(tfrecord_file):
    try:
        for record in tf.data.TFRecordDataset(tfrecord_file):
            # Attempt to parse the record
            _ = tf.train.Example.FromString(record.numpy())
    except tf.errors.DataLossError as e:
        print(f"DataLossError encountered: {e}")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return True
    return False

# Replace with your TFRecord file paths 


tfrecord_files = [
    r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\annotations\test.record',
    r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\annotations\train.record',
    r'boo12\TFODC\Tensorflow\workspace\annotations\TrainMerged.tfrecord',
]

for tfrecord_file in tfrecord_files:
  if is_tfrecord_corrupted(tfrecord_file):
      print(f"The TFRecord file {tfrecord_file} is corrupted.")
  else:
      print(f"The TFRecord file {tfrecord_file} is fine.")