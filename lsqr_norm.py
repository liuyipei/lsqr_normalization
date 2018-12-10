import tensorflow as tf
# import absl # todo use absl.logging

def weighted_update(var, new, momentum=.9, return_assigned_value=False, recast=False, assign_name=None):
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
          epsilon=1e-5, l2_regularizer=1e-3, affine_channels='afterall', residuals_norm1=True,
          marginalize='BHW', mean_variance_path=False):
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
    assert affine_channels in ['afterall', 'None']
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
        name_dict = dict()
    for k in ['lsq_beta', 'affine_b', 'affine_c', 'switch_logits']:
        if k not in name_dict.keys():
            name_dict[k] = k    

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
        dtype=tf.float32, 
        shape=[G, j, r], 
        initializer=tf.zeros_initializer(),
        trainable=False)

    
    x_BHWGc = tf.reshape(x_BHWC, [B, H, W, G, c]) 
    x_BHWGr = x_BHWGc[:, :, :, :, :r] # the first r channels are to have their residuals taken
    x_BHWGi = x_BHWGc[:, :, :, :, r:]
    x_BHW_Gi = tf.reshape(x_BHWGi, [B, H, W, G * i])
    if mean_variance_path:
        mean_Gi, var_Gi = switch_normalization_bn(x_BHW_Gi, C=G * i,
            training=training, momentum=momentum) # axis =3 assumed. no affine after
        z_BHW_Gi = (x_BHW_Gi - mean_Gi) * tf.math.rsqrt(var_Gi + epsilon)
    else:
        z_BHW_Gi = tf.layers.batch_normalization(
            x_BHW_Gi, axis=3, training=training,
            epsilon=epsilon, fused=True, renorm=False,
            center=False, scale=False)

    z_BHWGi = tf.reshape(z_BHW_Gi, [B, H, W, G, i])
    z_BHWGj = tf.concat([z_BHWGi, tf.ones([B, H, W, G, 1], dtype=tf.float32)], axis=4)

    z_GBHWj = tf.transpose(z_BHWGj, [3, 0, 1, 2, 4])
    x_GBHWr = tf.transpose(x_BHWGr, [3, 0, 1, 2, 4])
    def make_yhat(marginalize):
        if marginalize in ['BHW', 'runBHW']:
            z_G_BHW_j = tf.reshape(z_GBHWj, [G, B * H * W, j])
            x_G_BHW_r = tf.reshape(x_GBHWr, [G, B * H * W, r])
            curr_lsq_beta_Gjr = tf.linalg.lstsq(z_G_BHW_j, x_G_BHW_r, fast=True, l2_regularizer=l2_regularizer)
            update_beta = weighted_update(var_lsq_beta_Gjr, curr_lsq_beta_Gjr,
                                          momentum = momentum, return_assigned_value=False,
                                          recast=False, assign_name='assign_beta')
            if training and marginalize == 'BHW':
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_beta)
                lsq_beta_Gjr = curr_lsq_beta_Gjr
            else:
                lsq_beta_Gjr = tf.cast(var_lsq_beta_Gjr, tf.float32)
            yhat_BHWGr = tf.einsum('bhwgj,gjr->bhwgr', z_BHWGj , lsq_beta_Gjr)
        elif marginalize == 'HW':
            z_GB_HW_j = tf.reshape(z_GBHWj, [G, B, H * W, j])
            x_GB_HW_r = tf.reshape(x_GBHWr, [G, B, H * W, r])
            curr_lsq_beta_GBjr = tf.linalg.lstsq(z_GB_HW_j, x_GB_HW_r, fast=True, l2_regularizer=l2_regularizer)
            yhat_BHWGr = tf.einsum('bhwgj,gbjr->bhwgr', z_BHWGj , curr_lsq_beta_GBjr)
        #elif marginalize == 'sw-BHW-HW':
        #    raise NotImplementedError()
        else:
            raise ValueError()
        return yhat_BHWGr

    print('marginalize {} mean_variance_path {}'.format(marginalize, mean_variance_path))
    if marginalize in ['BHW', 'runBHW', 'HW']:
        yhat_BHWGr = make_yhat(marginalize)
    elif marginalize == 'sw-BHW-HW':
        yhat0_BHWGr = make_yhat('BHW') # like batch norm
        yhat1_BHWGr = make_yhat('HW')  # like instance norm
        switch_logits = tf.get_variable(
            name=name_dict['switch_logits'],  dtype=tf.float32,  shape=[2], 
            initializer=tf.zeros_initializer(), trainable=True)
        switch_probs = tf.nn.softmax(switch_logits)
        yhat_BHWGr = yhat0_BHWGr * switch_probs[0] + yhat1_BHWGr * switch_probs[1]
    
    residuals_BHWGr = x_BHWGr - yhat_BHWGr
    if residuals_norm1:
        residuals_BHW_Gr = tf.reshape(residuals_BHWGr, [B, H, W, G * r ]) # has mean 0 within each of the G * r channels
        residuals_BHW_Gr = tf.layers.batch_normalization(
            residuals_BHW_Gr, axis=3, center=False, scale=False, training=training, epsilon=epsilon,
            fused=True, renorm=False,)
        residuals_BHWGr = tf.reshape(residuals_BHW_Gr, [B, H, W, G, r ])
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
    if mean_variance_path:
        mean_BHWGi = tf.zeros(shape=[B, H, W, G, i]) + tf.reshape(mean_Gi, [G, i]) # broadcast
        mean_concat_BHWGc = tf.concat([yhat_BHWGr, mean_BHWGi], axis=4)
        mean_concat_BHWC = tf.reshape(mean_concat_BHWGc, [B, H, W, C])
        var_BHWGr = residuals_BHWGr ** 2
        var_BHWGi = tf.zeros(shape=[B, H, W, G, i]) + tf.reshape(var_Gi, [G, i])
        var_concat_BHWGc = tf.concat([var_BHWGr, var_BHWGi], axis=4)
        var_concat_BHWC = tf.reshape(var_concat_BHWGc, [B, H, W, C])
        helper_dict['mean_concat_BHWC'] = mean_concat_BHWC
        helper_dict['var_concat_BHWC'] = var_concat_BHWC

    if affine_channels == 'afterall':
        var_affine_b = tf.get_variable(
            name=name_dict.get('affine_b', 'affine_b'),  dtype=tf.float32,  shape=[C], 
            initializer=tf.zeros_initializer(), trainable=True)
        var_affine_c = tf.get_variable(
            name=name_dict.get('affine_c', 'affine_c'),  dtype=tf.float32,  shape=[C], 
            initializer=tf.ones_initializer(), trainable=True)
        z_BHWC = z_BHWC * var_affine_c + var_affine_b

    if data_format == 'channels_last':
        z_out = tf.identity(z_BHWC)
    elif data_format == 'channels_first': # BHWC -> BCHW  
        z_out = tf.transpose(z_BHWC, [0, 3, 1, 2])
    else:
        raise ValueError()

    return z_out, helper_dict



def as_BHWC(x, data_format='channels_last'):
    if data_format == 'channels_last':
        x_BHWC = tf.identity(x)
    elif data_format == 'channels_first': # BCHW -> BHWC
        x_BHWC = tf.transpose(x, [0, 2, 3, 1])
    else:
        raise ValueError()
    x_shape = tf.shape(x_BHWC)
    x_shape_list = x_BHWC.get_shape().as_list()
    B = x_shape[0]
    H = x_shape[1]
    W = x_shape[2]
    C = x_shape_list[3] 
    
    return x_BHWC, [B, H, W, C]



"""
explanatory_pinv_BHWGi = get_pinv_BHWGi(x_BHWGi, l2_regularizer)
z_out, helper_dict = shared_lsqrn(x_BHWC, explanatory_pinv_BHWGi, training=training, data_format=data_format)
"""



def get_pinv_BHWGi(x_BHWGi, l2_regularizer=1e-2, marginalize='BHW', 
                   mult_trace_to_l2_reg=None, floor_trace_to_l2_reg=1e-2):
    i = x_BHWGi.get_shape().as_list()[4] # known: number of explanatory (independent) channels in the regression
    reg_matrix = tf.eye(i) * l2_regularizer

    if marginalize == 'BHW':
        gram_Gii = tf.einsum('bhwgi,bhwgj->gij', x_BHWGi, x_BHWGi)
        if mult_trace_to_l2_reg is not None:
            print('mult_trace_to_l2_reg', mult_trace_to_l2_reg)
            trace_G11 = tf.expand_dims(tf.expand_dims(tf.linalg.trace(gram_Gii), axis=-1), axis=-1)
            trace_G11 = tf.maximum(floor_trace_to_l2_reg, trace_G11)
            reg_matrix *= trace_G11 * mult_trace_to_l2_reg
        gram_Gii += reg_matrix
        graminv_Gii = tf.linalg.inv(gram_Gii)
        x_pinv_BHWGi = tf.einsum('bhwgi,gij->bhwgj', x_BHWGi, graminv_Gii)
    elif marginalize == 'HW':
        gram_BGii = tf.einsum('bhwgi,bhwgj->bgij', x_BHWGi, x_BHWGi)
        if mult_trace_to_l2_reg is not None:
            print('mult_trace_to_l2_reg', mult_trace_to_l2_reg)
            trace_BG11 = tf.expand_dims(tf.expand_dims(tf.linalg.trace(gram_BGii), axis=-1), axis=-1)
            trace_BG11 = tf.maximum(floor_trace_to_l2_reg, trace_BG11)
            reg_matrix *= trace_BG11 * mult_trace_to_l2_reg
        gram_BGii += reg_matrix
        graminv_BGii = tf.linalg.inv(gram_BGii)
        x_pinv_BHWGi = tf.einsum('bhwgi,bgij->bhwgj', x_BHWGi, graminv_BGii)
    else:
        raise ValueError()
    return x_pinv_BHWGi


def shared_lsqrn(x, G, explanatory_pinv_BHWGi, BHWC=None, name_dict=None, training=False, momentum=.99, data_format='channels_last', 
                 epsilon=1e-5, bn_residuals=True, center_bn_residuals=True, scale_bn_residuals=True, marginalize='BHW', mean_variance_path=False):
    """
    """
    x_BHWC, [B, H, W, C] = as_BHWC(x, data_format)

    if name_dict is None:
        name_dict = {}
    for k in ['lsq_beta', 'affine_b', 'affine_c', 'switch_logits']:
        if k not in name_dict.keys():
            name_dict[k] = k
    
    i = explanatory_pinv_BHWGi.get_shape().as_list()[4]
    c  = C // G
    assert c * G == C
    print(name_dict['lsq_beta'], [G, i, c])
    print('shared_lsqrn: G = {}. i = {}. c = {}.'.format(G, i, c))
    var_lsq_beta_Gic = tf.get_variable(
        name=name_dict['lsq_beta'], 
        dtype=tf.float32, 
        shape=[G, i, c], 
        initializer=tf.zeros_initializer(),
        trainable=False)
    
    
    x_BHWGc = tf.reshape(x_BHWC, [B, H, W, G, c]) 
    def make_yhat(marginalize):
        if marginalize in ['BHW', 'runBHW']:
            print(explanatory_pinv_BHWGi, x_BHWGc)
            curr_lsq_beta_Gic = tf.einsum('bhwgi,bhwgc->gic', explanatory_pinv_BHWGi, x_BHWGc)  # tf.linalg.lstsq(z_G_BHW_j, x_G_BHW_r, fast=True, l2_regularizer=l2_regularizer)
            update_beta = weighted_update(var_lsq_beta_Gic, curr_lsq_beta_Gic,
                                          momentum = momentum, return_assigned_value=False,
                                          recast=False, assign_name='assign_beta')
            if training and marginalize == 'BHW':
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_beta)
                lsq_beta_Gic = curr_lsq_beta_Gic
            else:
                lsq_beta_Gic = tf.cast(var_lsq_beta_Gic, tf.float32)
            yhat_BHWGc = tf.einsum('bhwgi,gic->bhwgc', explanatory_pinv_BHWGi, lsq_beta_Gic)
        elif marginalize == 'HW':
            curr_lsq_beta_Gic = tf.einsum('bhwgi,bhwgc->gic', explanatory_pinv_BHWGi, x_BHWGc)  # tf.linalg.lstsq(z_G_BHW_j, x_G_BHW_r, fast=True, l2_regularizer=l2_regularizer)
            lsq_beta_Gic = curr_lsq_beta_Gic
            yhat_BHWGc = tf.einsum('bhwgi,gic->bhwgc', explanatory_pinv_BHWGi, lsq_beta_Gic)
        else:
            raise ValueError()
        return yhat_BHWGc

    print('marginalize {} mean_variance_path {}'.format(marginalize, mean_variance_path))
    if marginalize in ['BHW', 'runBHW', 'HW']:
        yhat_BHWGc = make_yhat(marginalize)
    elif marginalize == 'sw-BHW-HW':
        yhat0_BHWGc = make_yhat('BHW') # like batch norm
        yhat1_BHWGc = make_yhat('HW')  # like instance norm
        switch_logits = tf.get_variable(
            name=name_dict['switch_logits'],  dtype=tf.float32,  shape=[2], 
            initializer=tf.zeros_initializer(), trainable=True)
        switch_probs = tf.nn.softmax(switch_logits)
        yhat_BHWGr = yhat0_BHWGc * switch_probs[0] + yhat1_BHWGc * switch_probs[1]
    
    yhat_BHWC = tf.reshape(yhat_BHWGc, [B, H, W, C])
    residuals_BHWC = x_BHWC - yhat_BHWC
    if bn_residuals:
        z_BHWC = tf.layers.batch_normalization(
            residuals_BHWC, axis=3, center=center_bn_residuals, scale=scale_bn_residuals, training=training, epsilon=epsilon,
            fused=True, renorm=False,)
    else:
        z_BHWC = tf.identity(residuals_BHWC)

    helper_dict={
        'x_BHWGc': x_BHWGc,
        'residuals_BHWC': residuals_BHWC,
        'z_BHWC': z_BHWC,
        'name_dict': name_dict}

    if mean_variance_path:
        helper_dict['mean_concat_BHWC'] = yhat_BHWGr
        helper_dict['var_concat_BHWC'] = residuals_BHWC ** 2

    if data_format == 'channels_last':
        z_out = tf.identity(z_BHWC)
    elif data_format == 'channels_first': # BHWC -> BCHW  
        z_out = tf.transpose(z_BHWC, [0, 3, 1, 2])
    else:
        raise ValueError()

    return z_out, helper_dict

####

def switch_normalization(x, training, momentum=.9, epsilon=1e-3, data_format='channels_last', center=True, scale=True,
                         bn_treatment='base', Gr=None):
    assert data_format in ['channels_first', 'channels_last']
    if data_format == 'channels_first':
        x_BHWC = tf.transpose(x, [0, 2, 3, 1])
    else:
        x_BHWC = x        

    C = x_BHWC.get_shape().as_list()[3]
    if bn_treatment == 'base':
        mean_bn, var_bn = switch_normalization_bn(x_BHWC, C, training=training, momentum=momentum)
    elif bn_treatment == 'lsqrn': #(x_BHWC, marginalize, traning, momentum, epsilon)
        assert Gr is not None
        mean_bn, var_bn = switch_normalization_lsqrn(
            x_BHWC, Gr=Gr, marginalize='BHW',
            training=training, momentum=momentum, epsilon=1e-3)
    else:
        raise ValueError()
    mean_in, var_in = switch_normalization_in(x_BHWC)
    mean_ln, var_ln = switch_normalization_ln(x_BHWC)
    switch_logits = tf.get_variable(
        name='switch_logits_BIL',  dtype=tf.float32,  shape=[3], 
        initializer=tf.zeros_initializer(), trainable=True)
    p = tf.nn.softmax(switch_logits, name='switch_probs_BIL')
    mean = p[0] * mean_bn[0] + p[1] * mean_in + p[2] * mean_ln
    var = p[0] * var_bn[0] + p[1] * var_in + p[2] * var_ln
    inv_stddev = tf.math.rsqrt(epsilon + var)
    z_BHWC = (x_BHWC - mean) * inv_stddev

    if scale:
        var_affine_c = tf.get_variable(
            name='switch_c',  dtype=tf.float32,  shape=[C], 
            initializer=tf.ones_initializer(), trainable=True)
        z_BHWC = z_BHWC * var_affine_c
    if center:
        var_affine_b = tf.get_variable(
            name='switch_b',  dtype=tf.float32,  shape=[C], 
            initializer=tf.zeros_initializer(), trainable=True)
        z_BHWC = z_BHWC + var_affine_b

    if data_format == 'channels_first':
        z_BCHW = tf.transpose(z_BHWC, [0, 3, 1, 2])
        return z_BCHW, dict()
    else:
        return z_BHWC, dict()

def switch_normalization_bn(x_BHWC, C, training, momentum):
    var_mean = tf.get_variable(
        name='swn_bn_running_mean', 
        dtype=tf.float32, 
        shape=[C], 
        initializer=tf.zeros_initializer(),
        trainable=False)
    var_variance = tf.get_variable(
        name='swn_bn_running_variance', 
        dtype=tf.float32, 
        shape=[C], 
        initializer=tf.ones_initializer(),
        trainable=False)
    if training:
        mean, variance = tf.nn.moments(x_BHWC, axes=[0, 1, 2])
        update_mean = weighted_update(var_mean, mean,
                                      momentum = momentum, return_assigned_value=False,
                                      recast=False, assign_name='update_mean')
        update_variance = weighted_update(var_variance, variance,
                                          momentum = momentum, return_assigned_value=False,
                                          recast=False, assign_name='update_variance')
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)
    else:
        mean, variance = var_mean, var_variance
    return tf.reshape(mean, [1, 1, 1, C]), tf.reshape(variance, [1, 1, 1, C])

def switch_normalization_in(x_BHWC):
     mean, variance = tf.nn.moments(x_BHWC, keep_dims=True, axes=[1, 2])
     return mean, variance # [B, 1, 1, ]

def switch_normalization_ln(x_BHWC):
     mean, variance = tf.nn.moments(x_BHWC, keep_dims=True, axes=[1, 2, 3])
     return mean, variance

def switch_normalization_lsqrn(x_BHWC, Gr, marginalize, training, momentum, epsilon):
     _, helper_dict = lsqrn(x_BHWC, Gr=Gr, BHWC=None, name_dict=None, training=training, momentum=momentum,
          data_format='channels_last',  epsilon=epsilon, l2_regularizer=1e-3, affine_channels='afterall',
          residuals_norm1=False, marginalize='BHW', mean_variance_path=True)
     mean_concat_BHWC = helper_dict['mean_concat_BHWC']
     var_concat_BHWC = helper_dict['var_concat_BHWC']
     return mean_concat_BHWC, var_concat_BHWC
