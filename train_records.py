
#created on 2018/4/2
#anthor:

import os
import numpy as np
import tensorflow as tf
import model
import creat_records as cr

N_CLASSES = 2
IMG_W = 64 #resize the image, if the input image is too large, training will be very slow.  
IMG_H = 64
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 500 #with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 #with current parameters, it is suggested to use learning rate<0.0001


def run_training1():

    logs_train_dir = "D:/sunjiankang/tensorflow_model/" #*********************************need to modify**************************************
    
    tfrecords_file = "D:/sunjiankang/tensorflow_model/dogCat_model/a.tfrecords" #*********************************need to modify**************************************
    train_batch, train_label_batch = cr.read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
    train_batch = tf.cast(train_batch, dtype=tf.float32) #数据格式转换
    train_label_batch = tf.cast(train_label_batch, dtype=tf.int64)

    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)

    
    summary_op = tf.summary.merge_all() #获取所有监测的操作
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)#生成一个写日志的writer，并将当前的计算图写入日志
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
            if step % 10 == 0:
                print("**********************")
                print("Step %d, train loss = %.5f, train accuracy = %.2f%%" %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op) #获取监测结果
                train_writer.add_summary(summary_str, step) #写入文件
                

            if step % 2000 == 0 or (step +1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached")

    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()



run_training1()


