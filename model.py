import tensorflow as tf 


#Build the model 
    #Args: 
        #images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels] 
    #Returns: 
        #output tensor with the computed logits, float, [batch_size, n_classes] 

def inference(images, batch_size, n_classes):
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights", 
                                   shape=[3, 3, 3, 16], 
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        biases = tf.get_variable("biases",
                                  shape=[16],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))

        conv =tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    


    #pool1 and norm1
    #ksize=[1, height, width, 1], strides=[1, stride, stride, 1]
    with tf.variable_scope("poolint1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name="norm1")

    #conv2
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                   shape=[3, 3, 16, 16],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
                                   
        biases = tf.get_variable("biases",
                                  shape=[16],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    #pool2 and norm2
    with tf.variable_scope("poolinh2_lrn") as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name="norm2")
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME", name="pooling2")

    #local3
    with tf.variable_scope("local3") as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value# 动态获取tensor大小的方法
        weights = tf.get_variable("weights", 
                                   shape=[dim, 128],#输入向量长为dim，输出向量长为128
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))

        biases = tf.get_variable("biases", 
                                  shape=[128],
                                  dtype=tf.float32, 
                                  initializer=tf.constant_initializer(0.1))

        local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)#????????????

    #local4
    with tf.variable_scope("local4") as scope:
        weights = tf.get_variable("weights",
                                   shape = [128, 128],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                shape=[128],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights)+biases, name="local4")

    #softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("softmax_linear",
                                   shape=[128, n_classes],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                  shape=[n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name="softmax_linear")#a*b+c

    return softmax_linear    


#Compute loss from logits and labels 
    #Args: 
        #logits: logits tensor, float, [batch_size, n_classes] 
        #labels: label tensor, tf.int32, [batch_size] 
         
    #Returns: 
        #loss tensor of float type 

def losses(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="xentropy_per_example")#计算交叉熵来刻画预测值与真实值的误差
        loss = tf.reduce_mean(cross_entropy, name="loss")#计算当前batch所有样例的交叉熵的平均值
        tf.summary.scalar(scope.name+"/loss", loss)
    return loss



def training(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#AdamOptimizer控制学习速度,通过使用动量（参数的移动平均数）来改善传统梯度下降，促进超参数动态调整。
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy =tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+"/accuracy", accuracy)
    return accuracy

                                                                                  


    



