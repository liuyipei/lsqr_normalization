import unittest
import tensorflow as tf
import numpy as np
import lsqr_norm


class TestStringMethods(unittest.TestCase):

        
    def test_lsqrn(self,):
        B, H, W, C = [2, 11, 13, 28]
        G, r = 4, 3
        G = 4
        c = C // G 
        r = 3
        i = c - r
        j = i + 1
        training = True
        momentum = .9
        print("ijcr:", i, j, c, r)

        assert c * G == C

        x_BHWC = tf.placeholder(shape=[B, H, W, C], dtype=tf.float32)
        x_np_rand = np.random.random([B, H, W, C])
        z_BHWC, helper_dict = lsqr_norm.lsqrn(x_BHWC, Gr=[G, r], BHWC=[B, H, W, C],
            name_dict = {'lsq_beta': 'lsq_beta0'}, training=training, momentum=momentum,
            data_format='channel_last')
        # two from layers batch normalization, one from our update
        assert len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) == 3

        z1_BCHW, helper_dict = lsqr_norm.lsqrn(tf.transpose(x_BHWC, [0, 3, 1, 2]), 
            Gr=[G, r], BHWC=[B, H, W, C],
            name_dict = {'lsq_beta': 'lsq_beta1'}, training=training, momentum=momentum,
            data_format='channels_first')
        z1_BHWC = tf.transpose(z1_BCHW, [0, 2, 3, 1])

        x_BHWGr = helper_dict['x_BHWGr']
        x_BHWGi = helper_dict['x_BHWGi']
        residuals_BHWGr = helper_dict['residuals_BHWGr']
        # x_BHWGj = helper_dict['x_BHWGj']

        s = tf.Session()
        init_all = tf.global_variables_initializer()
        s.run(init_all)
        feed_dict = {x_BHWC: x_np_rand}
        np_resids = s.run(residuals_BHWGr, feed_dict=feed_dict)
        np_x_BHWGr = s.run(x_BHWGr, {x_BHWC: x_np_rand})
        assert np.linalg.norm(np_resids) < np.linalg.norm(np_x_BHWGr)
        
        # attempt to collect gradients
        loss = tf.reduce_sum(residuals_BHWGr ** 2)
        gradients_x = tf.gradients(loss, x_BHWC)
        gradients_x_BHWGr = tf.gradients(loss, x_BHWGr)
        gradients_x_BHWGi = tf.gradients(loss, x_BHWGi)
        np_resids = s.run([gradients_x, gradients_x_BHWGi, gradients_x_BHWGr],
                          {x_BHWC: x_np_rand}) # attempt to get gradients

        
        
        # try to run a train opo
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_op = optimizer.minimize(loss)
        s.run(train_op, feed_dict=feed_dict)

        zshape0, zshape1 = s.run([tf.shape(z_BHWC), tf.shape(z1_BHWC)])
        print(zshape0, zshape1)
        assert np.all(zshape0 == zshape1)


if __name__ == '__main__':
    unittest.main()