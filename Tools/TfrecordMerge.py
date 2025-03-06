

import tensorflow as tf


def write_to_tfrecord(serialized_examples, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for serialized_example in serialized_examples:
            writer.write(serialized_example)


def merge_tfrecords(file1, file2, output_file):
    serialized_examples = []

    # Read examples from first file
    for record in tf.data.TFRecordDataset(file1):
        serialized_examples.append(record.numpy())

    # Read examples from second file
    for record in tf.data.TFRecordDataset(file2):
        serialized_examples.append(record.numpy())
        
    

    # Write all examples to a new TFRecord file
    write_to_tfrecord(serialized_examples, output_file)



def merge_three_tfrecords(file1, file2, file3, output_file):
    serialized_examples = []

    # Read examples from first file
    for record in tf.data.TFRecordDataset(file1):
        serialized_examples.append(record.numpy())

    # Read examples from second file
    for record in tf.data.TFRecordDataset(file2):
        serialized_examples.append(record.numpy())
        
    # Read examples from third file
    for record in tf.data.TFRecordDataset(file3):
        serialized_examples.append(record.numpy())

    # Write all examples to a new TFRecord file
    write_to_tfrecord(serialized_examples, output_file)




#  V3I dataset combine val and train 
file1 = r'boo12\TFODC\Tensorflow\workspace\annotations\train.record'
file2 = r'boo12\TFODC\Tensorflow\workspace\annotations\valid.tfrecord'
output_file = r'boo12\TFODC\Tensorflow\workspace\annotations\merged.tfrecord'


#merge_tfrecords(file1, file2, output_file)



# VOC dataset combine val and train 
file1 = r'C:\Users\besto\Downloads\VOC PCB.v1-tensorflow_defect_dataset.tfrecord\valid\Defects.tfrecord'
file2 = r'C:\Users\besto\Downloads\VOC PCB.v1-tensorflow_defect_dataset.tfrecord\train\Defects.tfrecord'
output_file = r'C:\Users\besto\Downloads\VOC PCB.v1-tensorflow_defect_dataset.tfrecord\train\VocTrainMerged.tfrecord'


#merge_tfrecords(file1, file2, output_file)




#New combine train dataset
file1 = r"C:\Users\besto\Downloads\VOC PCB.v1-tensorflow_defect_dataset.tfrecord\Combined datasets\VocTrainMerged.tfrecord"
file2 = r"C:\Users\besto\Downloads\VOC PCB.v1-tensorflow_defect_dataset.tfrecord\Combined datasets\TrainMerged.tfrecord"
output_file = r"C:\Users\besto\Downloads\VOC PCB.v1-tensorflow_defect_dataset.tfrecord\Combined datasets\CombinedTrain.tfrecord"


#merge_tfrecords(file1, file2, output_file)

#New combine test dataset
file1 = r"C:\Users\besto\Downloads\VOC PCB.v1-tensorflow_defect_dataset.tfrecord\Combined datasets\Test.tfrecord"
file2 = r"C:\Users\besto\Downloads\VOC PCB.v1-tensorflow_defect_dataset.tfrecord\Combined datasets\test.tfrecord"
output_file = r"C:\Users\besto\Downloads\VOC PCB.v1-tensorflow_defect_dataset.tfrecord\Combined datasets\CombinedTest.tfrecord"

#merge_tfrecords(file1, file2, output_file)
# i combined the dataset twice for testing






#merge new v1i  test dataset
file1 = r"C:\Users\besto\Downloads\Final dataset\PCB-defects-valid.tfrecord"
file2 = r"C:\Users\besto\Downloads\Final dataset\PCB-defects-test.tfrecord"
output_file = r"C:\Users\besto\Downloads\Final dataset\V1ItestMerged.tfrecord"
#added val to test as test size was small

#merge_tfrecords(file1, file2, output_file)



# New final dataset train dataset
file1 = r"C:\Users\besto\Downloads\Final dataset\Train\V1ITrainMerged.tfrecord"
file2 = r"C:\Users\besto\Downloads\Final dataset\Train\VOCTrainMerged.tfrecord"
file3 = r"C:\Users\besto\Downloads\Final dataset\Train\v3iTrainMerged.tfrecord"
output_file = r"C:\Users\besto\Downloads\Final dataset\Train\FinalTrain.tfrecord"
merge_three_tfrecords(file1, file2, file3, output_file)

# New final dataset test dataset
file1 = r"C:\Users\besto\Downloads\Final dataset\Test\V1ItestMerged.tfrecord"
file2 = r"C:\Users\besto\Downloads\Final dataset\Test\VOCTest.tfrecord"
file3 = r"C:\Users\besto\Downloads\Final dataset\Test\v3itest.record"
output_file = r"C:\Users\besto\Downloads\Final dataset\Test\FinalTest.record"
merge_three_tfrecords(file1, file2, file3, output_file)