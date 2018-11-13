import tensorflow as tf
# import absl # todo use absl.logging

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

def lsqrn(x, Gr, BHWC=None, name_dict=None, training=False, momentum=.99, data_format='channels_last', 
          epsilon=1e-5, l2_regularizer=1e-3, affine_channels='afterall'):
    """
    Parameters:
    ===========
    x: tf.Tensor()
        input float tensor known shape BHWC or BCHW, depending on data_format
    BHWC: [int, int, int int]
        shape of x_BHWC. Must all be positive integers
    Gr: [int, int]
        G is the number of groups. G must divide C
        r is the number of residual channels to compute for each group. r must be less than or equal to G
    TODO: Support i = 0 or r = 0
    """
    assert affine_channels in ['indep', 'afterall']
    if data_format == 'channels_last':
        x_BHWC = tf.identity(x)
    elif data_format == 'channels_first': # BCHW -> BHWC
        x_BHWC = tf.transpose(x, [0, 2, 3, 1])
    else:
        raise ValueError()

    if BHWC is not None:
        [B, H, W, C] = BHWC
    else:
        x_shape = tf.shape(x_BHWC)
        x_shape_list = x_BHWC.get_shape().as_list()
        B = x_shape[0]
        H = x_shape[1]
        W = x_shape[2]
        C = x_shape_list[3] 

    [G, r] = Gr
    c = C // G 
    
    i = c - r
    j = i + 1

    if name_dict is None:
        name_dict = {'lsq_beta': 'lsq_beta',
                     'affine_b': 'affine_b',
                     'affine_c': 'affine_c'}
    print(name_dict['lsq_beta'], [G, j, r])
    print('lsqr_norm: i = {}. r = {}.'.format(i, r))
    if i <= 0 or r <= 0:
        print("reverting to basic BN")
        return tf.layers.batch_normalization(
            x_BHWC, axis=3, training=training, epsilon=epsilon,
            fused=True, renorm=False,
            ), {'name_dict': name_dict}
    assert i > 0
    var_lsq_beta_Gjr = tf.get_variable(
        name=name_dict['lsq_beta'], 
        dtype=tf.float64, 
        shape=[G, j, r], 
        initializer=tf.zeros_initializer(),
        trainable=False)

    
    x_BHWGc = tf.reshape(x_BHWC, [B, H, W, G, c]) 
    x_BHWGr = x_BHWGc[:, :, :, :, :r] # the first r channels are to have their residuals taken
    x_BHWGi = x_BHWGc[:, :, :, :, r:]
    x_BHW_Gi = tf.reshape(x_BHWGi, [B, H, W, G * i])
    z_BHW_Gi = tf.layers.batch_normalization(x_BHW_Gi, axis=3, training=training,
        epsilon=epsilon, fused=True, renorm=False,
        center=affine_channels=='indep',
        scale=affine_channels=='indep')
    z_BHWGi = tf.reshape(z_BHW_Gi, [B, H, W, G, i])
    z_BHWGj = tf.concat([z_BHWGi, tf.ones([B, H, W, G, 1], dtype=tf.float32)], axis=4)

    # hat matrix:  X (X* X)^-1 X*
    # cap matrix: (X* X)^-1 X*
    # lsq_beta: (X* X)^-1 X* y
    z_G_BHW_j = tf.reshape(tf.transpose(z_BHWGj, [3, 0, 1, 2, 4]), [G, B * H * W, j])
    x_G_BHW_r = tf.reshape(tf.transpose(x_BHWGr, [3, 0, 1, 2, 4]), [G, B * H * W, r])
    curr_lsq_beta_Gjr = tf.linalg.lstsq(z_G_BHW_j, x_G_BHW_r, fast=True, l2_regularizer=l2_regularizer)
    update_beta = weighted_update(var_lsq_beta_Gjr, curr_lsq_beta_Gjr,
                                  momentum = momentum, return_assigned_value=False,
                                  recast=True, assign_name='assign_beta')
    if training:
        lsq_beta_Gjr = curr_lsq_beta_Gjr
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_beta)
    else:
        lsq_beta_Gjr = tf.cast(var_lsq_beta_Gjr, tf.float32)

    yhat_BHWGr = tf.einsum('bhwgj,gjr->bhwgr', z_BHWGj , lsq_beta_Gjr)
    residuals_BHWGr = x_BHWGr - yhat_BHWGr
    z_BHWGc = tf.concat([residuals_BHWGr, z_BHWGi], axis=4)
    z_BHWC = tf.reshape(z_BHWGc, [B, H, W, C])
    helper_dict={
        'x_BHWGc': x_BHWGc,
        'x_BHWGr': x_BHWGr,
        'x_BHWGi': x_BHWGi,
        'x_BHW_Gi': x_BHW_Gi,
        'z_BHWGj': z_BHWGj,
        'z_BHW_Gi': z_BHW_Gi, 
        'residuals_BHWGr': residuals_BHWGr,
        'z_BHWGc': z_BHWGc,
        'z_BHWC': z_BHWC,
        'name_dict': name_dict}

    if affine_channels == 'afterall':
        var_affine_b = tf.get_variable(
            name=name_dict['affine_b'],  dtype=tf.float32,  shape=[C], 
            initializer=tf.zeros_initializer(), trainable=True)
        var_affine_c = tf.get_variable(
            name=name_dict['affine_c'],  dtype=tf.float32,  shape=[C], 
            initializer=tf.ones_initializer(), trainable=True)
        z_BHWC = z_BHWC * var_affine_c + var_affine_b

    if data_format == 'channels_last':
        z_out = tf.identity(z_BHWC)
    elif data_format == 'channels_first': # BHWC -> BCHW  
        z_out = tf.transpose(z_BHWC, [0, 3, 1, 2])
    else:
        raise ValueError()

    return z_out, helper_dict

    
