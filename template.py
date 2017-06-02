import tensorflow as tf

# 初始化变量和模型参数，定义训练闭环中的运算


def inference():
    # 计算推断模型在数据 X 上的输出，并将结果返回
    pass


def loss(X, Y):
    # 依据训练数据 X 和期望数据 Y 计算损失值
    pass


def inputs():
    # 读取或生成训练数据 X 和期望数据 Y
    pass


def train(total_loss):
    # 依据计算的总损失训练或调整模型参数
    pass


def evaluate(sess, X, Y):
    # 对训练得到的模型进行评估
    pass


# 在一个会话对象中启动数据流图，搭建流程
with tf.Session() as sess:

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])

        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))

        if step % 1000 == 0:
            saver.save(sess, 'my-model', global_step=step)

    saver.save(sess, 'my-model', global_step=training_steps)
