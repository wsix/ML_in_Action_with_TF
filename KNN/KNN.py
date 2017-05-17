import tensorflow as tf


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, dataSet, labels, metrix='l2'):
        self.dataSet = tf.constant(dataSet, dtype=tf.float32)
        self.labels = labels
        self.target_x = tf.placeholder(tf.float32, [1, int(self.dataSet.shape[-1])])
        self.dist = tf.reduce_sum(tf.square(tf.subtract(self.dataSet, self.target_x)), axis=1)

        if metrix == 'l1':
            self.dist = tf.reduce_sum(tf.abs(tf.subtract(self.dataSet, self.target_x)), axis=1)

        self.nn = tf.nn.top_k(-self.dist, self.k)
        self.graph = tf.get_default_graph()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def predict(self, inX):
        results = self.sess.run(self.nn, feed_dict={self.target_x: inX})
        indices = results.indices
        index_count = {}
        for index in indices:
            if self.labels[index] not in index_count:
                index_count[self.labels[index]] = 1
            else:
                index_count[self.labels[index]] += 1

        pred_label = ''
        max_count = 0
        for key in index_count:
            if index_count[key] > max_count:
                pred_label = key
                max_count = index_count[key]

        return pred_label
