import tensorflow as tf


def weighted_update(var, new, momentum=.9, return_assigned_value=False, recast=True, assign_name=None):
    recast_dtype = new.dtype
    if recast:
        new = tf.cast(new, var.dtype)
    decay = 1. - momentum
    update_delta = (var - new) * decay
    assigned_value = var - update_delta
    assign_op = tf.assign(var, assigned_value, name=assign_name)
    if return_assigned_value:
        if recast:
            assigned_value = tf.cast(assigned_value, recast_dtype)
        return assign_op, assigned_value
    else:
        return assign_op


def lsqrn(x_BHWC, BHWC, Gr, name_dict=None, training=False, momentum=.9):
    # todo, let B be -1
    #G = 4
    #r = 3
    [B, H, W, C] = BHWC
    [G, r] = Gr
    c = C // G 
    
    i = c - r
    j = i + 1
    assert i > 0

    if name_dict is None:
        name_dict = {'lsq_beta': 'lsq_beta'}
    var_lsq_beta_Gjr = tf.get_variable(
        name=name_dict['lsq_beta'], 
        dtype=tf.float64, 
        shape=[G, j, r], 
        initializer=tf.zeros_initializer(),
        trainable=False)

    
    x_BHWGc = tf.reshape(x_BHWC, [B, H, W, G, c]) 
    x_BHWGr = x_BHWGc[:, :, :, :, :r]
    x_BHWGi = x_BHWGc[:, :, :, :, r:]
    z_BHWGi = tf.layers.batch_normalization(x_BHWGi, axis=4,  training=training)
    z_BHWGj = tf.concat([z_BHWGi, tf.ones([B, H, W, G, 1], dtype=tf.float32)], axis=4)

    # hat matrix:  X (X* X)^-1 X*
    # cap matrix: (X* X)^-1 X*
    # lsq_beta: (X* X)^-1 X* y
    z_G_BHW_j = tf.reshape(tf.transpose(z_BHWGj, [3, 0, 1, 2, 4]), [G, B * H * W, j])
    x_G_BHW_r = tf.reshape(tf.transpose(x_BHWGr, [3, 0, 1, 2, 4]), [G, B * H * W, r])
    curr_lsq_beta_Gjr = tf.linalg.lstsq(z_G_BHW_j, x_G_BHW_r)
    update_beta = weighted_update(var_lsq_beta_Gjr, curr_lsq_beta_Gjr, momentum = momentum, return_assigned_value=False,
                                  recast=True, assign_name='assign_beta')
    if training:
        lsq_beta_Gjr = curr_lsq_beta_Gjr
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_beta)
    else:
        lsq_beta_Gjr = var_lsq_beta_Gjr

    yhat_BHWGr = tf.einsum('bhwgj,gjr->bhwgr', z_BHWGj , lsq_beta_Gjr)
    residuals_BHWGr = x_BHWGr - yhat_BHWGr
    tensors={
        'x_BHWGc': x_BHWGc,
        'x_BHWGr': x_BHWGr,
        'x_BHWGi': x_BHWGi,
        'x_BHWGj': x_BHWGj,

    }
    helper_dict = dict(name_dict=name_dict)
    return residuals_BHWGr, helper_dict

    
