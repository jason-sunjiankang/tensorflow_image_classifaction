import cv2   
import tensorflow as tf  
import numpy as np  
import os  
import matplotlib.pyplot as plt  
import skimage.io as io  
from skimage import transform

from PIL import Image  

# get image list and label list
def get_files(file_dir):

    cats = []  
    label_cats = []  
    dogs = []  
    label_dogs = [] 

    # classifation{"cat", "dog"}
    # get cat and dog image list
    for file in os.listdir(file_dir + "/" + "cats"): #*********************************need to modify**************************************
        cats.append(file_dir + "/" + "cats" + "/" + file)
        label_cats.append(0) 
    for file in os.listdir(file_dir + "/" + "dogs"): #*********************************need to modify**************************************
        dogs.append(file_dir + "/" + "dogs" + "/" + file)
        label_dogs.append(1)

    print("num cats: %d\n" %len(cats))#*********************************need to modify************************************** 
    print("num dogs: %d\n" %len(dogs))#*********************************need to modify************************************** 

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    # Use shuffle to disrupt the order
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]#effect?

    return image_list, label_list

# Wrapper for inserting int64 features into Example proto.
def int64_feature(value):  
  if not isinstance(value, list):  
    value = [value]  
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))  
  
def bytes_feature(value):  
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  


# convert all images and labels to one tfrecord file
def convert_to_tfrecord(images, labels, save_dir, name):

    filename = save_dir + name + ".tfrecords"
    
    n_samples =len(images)
    if len(labels) != n_samples:  
        raise ValueError('Images size %d does not match label size %d.' %(len(images), len(labels)))  
        
    writer = tf.python_io.TFRecordWriter(filename)
    print("\nconvert to tfrecord start.........")


    for i in np.arange(0, n_samples):
        try:
            image = cv2.imread(images[i])
            image = cv2.resize(image, (64, 64)) #*********************************need to modify**************************************

           
            #cv2.waitKey(1)
            #b,g,r = cv2.split(image)
            #image = cv2.merge([r,g,b])
            #cv2.imshow("image", image)

            # image and label converted to binary and saved to tfrecords
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={  
                            'label':int64_feature(label),  
                            'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e: 
            print('Could not read:', images[i])  
            print('error: %s' %e)  
            print('Skip it!\n')
    writer.close()
    print("\nconvert to tfrecord down.........")


# read and decode tfrecord file, generate (image, label) batches
# warning!!!!! when training return image_batch and label_batch
      # returns:
        # image:4D tensor - [batch_size, width, height, channel] 
        # label: 1D tensor - [batch_size]
# warning!!!!! when test the image decoded  from tfrecode, need return image_src and label

def read_and_decode(tfrecords_file, batch_size):
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(  
                                        serialized_example,  
                                        features={  
                                               'label': tf.FixedLenFeature([], tf.int64),  
                                               'image_raw': tf.FixedLenFeature([], tf.string),  
                                               })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    image_src = tf.reshape(image, [64, 64,3])  #*********************************need to modify**************************************
    label = tf.cast(img_features['label'], tf.float32)      
    image = tf.image.per_image_standardization(image_src)  
    image_batch, label_batch = tf.train.batch([image, label],  
                                                batch_size= batch_size,  
                                                num_threads= 64,   
                                                capacity = 2000)#*********************************need to modify**************************************
    
    return image_batch, tf.reshape(label_batch, [batch_size]) 
    #return image_src, label


      
   

#**************************************test********************************************
#****************code image to TFRecord and decode TFRecord to image*******************
#!!!!!warning：when testing, function read_and_decode() need to return image_src, label

# file_dir = "C:/Users/Jiankang/Desktop/sjk/tensorflow/tensorflowData/catDog/train1000"
# #file_dir = "D:/Re_train/image_data/inputdata"
# image_list, label_list = get_files(file_dir)
# convert_to_tfrecord(image_list, label_list, "D:/", "a")
 
# batch = read_and_decode("D:/a.tfrecords", 20)
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
# with tf.Session() as sess: #开始一个会话      
      
#         sess.run(init_op)
#         coord=tf.train.Coordinator()
#         threads= tf.train.start_queue_runners(coord=coord)    

#         for i in range(50):   
#             img, lab = sess.run(batch)#get image and label from session，function read_and_decode() need return image_src, label     
#             #img=Image.fromarray(example, 'RGB')#这里Image是之前提到的  
#             cv2.imshow("img",np.array(img, dtype=np.uint8))
#             print(lab)
#             cv2.waitKey()
                        
#         coord.request_stop()      
#         coord.join(threads)     
#         sess.close()   
        
    