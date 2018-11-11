import unittest
import tensorflow as tf
import numpy as np
import lsqr_norm
class TestStringMethods(unittest.TestCase):

        
    def test_lsqrn():
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
        residuals_BHWGr, helper_dict = lsqr_norm.lsqrn(x_BHWC, BHWC=[B, H, W, C], Gr=[G, r], 
            name_dict=None, training=training, momentum=momentum)
        x_BHWGr = helper_dict['x_BHWGr']
        x_BHWGi = helper_dict['x_BHWGi']
        # x_BHWGj = helper_dict['x_BHWGj']

        s = tf.Session()
        init_all = tf.global_variables_initializer()
        s.run(init_all)
        np_resids = s.run(residuals_BHWGr, {x_BHWC: x_np_rand})
        np_x_BHWGr = s.run(x_BHWGr, {x_BHWC: x_np_rand})
        print(np.linalg.norm(np_resids), np.linalg.norm(np_x_BHWGr))
        
        # attempt to collect gradients
        loss = tf.reduce_sum(residuals_BHWGr ** 2)
        gradients_x = tf.gradients(loss, x_BHWC)
        gradients_x_BHWGr = tf.gradients(loss, x_BHWGr)
        gradients_x_BHWGi = tf.gradients(loss, x_BHWGi)
        np_resids = s.run([gradients_x, gradients_x_BHWGi, gradients_x_BHWGr], {x_BHWC: x_np_rand}) # attempt to get gradients

        # two from layers batch normalization, one from our update
        assert len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) == 3
        
        # try to run a train opo
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_op = optimizer.minimize(loss)
        s.run(train_op)



if __name__ == '__main__':
    unittest.main()