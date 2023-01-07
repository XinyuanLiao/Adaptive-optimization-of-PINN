import math

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import Dao
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)  
ax1.set_xlabel('its')
ax2.set_xlabel('its')
ax3.set_xlabel('its')
ax1.set_ylabel('loss')
ax2.set_ylabel('lamda_1')
ax3.set_ylabel('lamda_2')

data = [[] for i in range(4)]
np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    it = 0

    def __init__(self, X, u, layers, lb, ub):
        self.lb = lb
        self.ub = ub

        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.u = u

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)
        
        # hyper-parameter a
        # adaptive activation parameter
        # scale factor n
        self.n = tf.constant([1.])
        self.a = tf.constant([1], dtype=tf.float32)

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x_tf, self.t_tf)
        
        # trainable param for adaptive loss function
        self.σ1 = tf.Variable([0.0], dtype=tf.float32)
        self.σ2 = tf.Variable([0.0], dtype=tf.float32)
        
        # Parameters to be estimated
        self.l1 = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.l2 = tf.reduce_mean(tf.square(self.f_pred))
        
        self.loss1 = self.likelihood_loss()
        self.loss = self.true_loss()

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000, 
                                                                         'maxfun': 50000, 
                                                                         'maxcor': 50,
                                                                         'maxls': 50,  
                                                                         'ftol': 1.0 * np.finfo(float).eps})  

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.train_op_Adam1 = self.optimizer_Adam.minimize(self.loss1)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def likelihood_loss(self):
        loss1 = tf.stop_gradient(self.l1)
        loss2 = tf.stop_gradient(self.l2)
        loss = tf.exp(-self.σ1) * loss1 + self.σ1 \
               + tf.exp(-self.σ2) * loss2 + self.σ2
        return loss

    def true_loss(self):
        w1 = tf.stop_gradient(self.σ1)
        w2 = tf.stop_gradient(self.σ2)
        return tf.exp(-w1) * self.l1 + tf.exp(-w2) * self.l2
        # return self.l1+self.l2

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(self.n * self.a * tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        lambda_1 = self.lambda_1
        lambda_2 = tf.exp(self.lambda_2)
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f

    def callback(self, loss, lambda_1, lambda_2, a, σ1, σ2):

        if loss < 0.1:
            data[0].append(self.it)
            data[1].append(loss)
            data[2].append(lambda_1)
            data[3].append(np.exp(lambda_2))
            print('It: %d,  Loss: %e,  l1: %.5f,  l2: %.5f,  a:%.5f,  σ1:%.5f,  σ2:%.5f' %
                  (self.it, loss, lambda_1, np.exp(lambda_2), a, σ1, σ2))
        self.it = self.it + 1

    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}

        start_time = time.time()
        for its in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            self.sess.run(self.train_op_Adam1, tf_dict)
            # Print
            loss_value = self.sess.run(self.loss, tf_dict)
            loss1_value = self.sess.run(self.loss1, tf_dict)
            w1 = self.sess.run(self.σ1)
            w2 = self.sess.run(self.σ2)
            lambda_1_value = self.sess.run(self.lambda_1)
            lambda_2_value = np.exp(self.sess.run(self.lambda_2))
            if (loss_value < 1):
                print('It: %d, Loss: %.3e, Loss1: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, σ1:%.5f, σ2:%.5f' %
                      (its, loss_value, loss1_value, lambda_1_value, lambda_2_value, w1, w2))
                data[0].append(its)
                data[1].append(loss_value)
                data[2].append(lambda_1_value)
                data[3].append(lambda_2_value)
            self.it = self.it + 1
        # self.optimizer.minimize(self.sess,
        #                         feed_dict=tf_dict,
        #                         fetches=[self.loss, self.lambda_1, self.lambda_2, self.a, self.σ1, self.σ2],
        #                         loss_callback=self.callback)
        elapsed = time.time() - start_time
        print('total time:%.2f' % elapsed)

    def predict(self, X_star):

        tf_dict = {self.x_tf: X_star[:, 0:1], self.t_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)

        return u_star, f_star


if __name__ == "__main__":
    N_u = 3000
    nu = 0.01 / np.pi
    layers = [2, 20, 20, 20, 20, 20, 20, 1]

    sql_domain = "select x,t,usol from burgers order by t,x"
    result_domain = Dao.loadSql(sql_domain)
    X_star = np.array(result_domain)[:, 0:2]
    u_star = np.array(result_domain)[:, -1]
    u_star = [u_star]
    u_star = np.array(u_star).T

    lb = X_star.min(0)
    ub = X_star.max(0)

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]

    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model.train(1000)

    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    lambda_2_value = np.exp(lambda_2_value)

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

    print('Error u: %e' % (error_u))
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))

    ax1.plot(data[0], data[1], c="red", label="loss")
    ax2.plot(data[0], data[2], c="blue", label="lamda_1")
    ax3.plot(data[0], data[3], c="green", label="lamda_2")
    plt.show()
