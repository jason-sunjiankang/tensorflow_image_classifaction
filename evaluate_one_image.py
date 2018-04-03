
#created on 2018/4/2
#anthor:

import os
import model
import numpy as np
import tensorflow as tf
import cv2



def get_one_image(test_image_dir):
   
    image = cv2.imread(test_image_dir)
    img=cv2.resize(image, (64, 64))
    #cv2.imshow("img",np.array(img, dtype=np.uint8))
    return img

def get_images(test_file_dir):
    test_images = []
    for file in os.listdir(test_file_dir):
        test_images.append(test_file_dir+file)
    return test_images


def evaluate_one_image(image_array):

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 64, 64, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[64, 64, 3])

        logs_train_dir = "D:/sunjiankang/tensorflow_model/dogCat_model/"

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints......")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loading success, global_step is %s" %global_step)
            else:
                print("No checkpoint file found")

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            # if max_index == 0:
            #     print("This is a cat with possibility %.6f" %prediction[:, 0])
            # else:
            #     print("This is a dog with possibility %.6f" %prediction[:, 1])

    return max_index


#******************************test one image*******************************
# test_image_dir = "C:/Users/Jiankang/Desktop/sjk/tensorflow/tensorflowData/catDog/test/10.jpg"  
# img = get_one_image(test_image_dir)
# cv2.imshow("img", img)
# cv2.waitKey()
# evaluate_one_image(img)  


#******************************test images*******************************
test_file_dir = "C:/Users/Jiankang/Desktop/sjk/tensorflow/tensorflowData/catDog/cats/"  
imgs = get_images(test_file_dir)

j = 0
#for i in range(0, len(imgs)):
for i in np.arange(len(imgs)):
    img = get_one_image(imgs[i])
    # cv2.imshow("img", img)
    # cv2.waitKey()
    cout = evaluate_one_image(img)
    if(cout == 0):
        j = j+1
    else:
        print(imgs[i])

accuracy = (float(j)/float(len(imgs)))*100
print("j=%d" %j)
print("Test accuracy %5f%%" %accuracy)

    
    
    