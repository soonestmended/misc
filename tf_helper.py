from keras import Input, Model, models, layers, losses, metrics, optimizers

kwargs = {
    'kernel_size': (1, 3, 3),
    'padding': 'same'
}

#conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
def conv(x, filters, strides, kernel_size = (1, 3, 3), padding = 'same'):
    return layers.Conv3D(filters = filters, strides = strides, kernel_size = kernel_size, padding = padding)(x)

# norm = lambda x : layers.BatchNormalization()(x)
def norm(x):
    return layers.BatchNormalization()(x)
    
# relu = lambda x : layers.ReLU()(x)
def relu(x):
    return layers.ReLU()(x)

# --- Define stride-1, stride-2 blocks
# norm_relu_conv11 = lambda filters, x : conv(relu(norm(x)), filters, strides=1)
def nrc1(filters, x):
    return conv(relu(norm(x)), filters, strides=1)

# norm_relu_conv22 = lambda filters, x : conv(relu(norm(x)), filters, strides=(1, 2, 2))
def nrc2(filters, x):
    return conv(relu(norm(x)), filters, strides=(1, 2, 2))

def match_dims(x, shape):
    # alter dimensions of x to match [shape]
    x_shape = x.shape.as_list()
#    print("x_shape", x_shape)
#    print("target shape", shape)
    if x_shape == shape:
        return x
    proj_filters = shape[4]
    proj_strides = int(x_shape[2] / shape[2])
    proj_kernel = (1, 1, 1) if proj_strides == 1 else (1, 3, 3)
    return layers.Conv3D(filters = shape[4], strides = (1, proj_strides, proj_strides), kernel_size = proj_kernel, padding = 'same')(x)

def inception(filters, x):
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides = 1, kernel_size=(1, 1, 1))))
    conv3 = lambda filters, x : relu(norm(conv(x, filters, strides = 1, kernel_size=(1, 3, 3))))
    conv5 = lambda filters, x : relu(norm(conv(x, filters, strides = 1, kernel_size=(1, 5, 5))))
    mpool = lambda x : layers.MaxPool3D(pool_size=(1, 3, 3), strides=1, padding='same')(x)
    
    filters = int(filters/4)
    p1 = conv1(filters, x)
    p2 = conv3(filters, x)
    p3 = conv5(filters, x)
    p4 = mpool(x)

    # --- Concatenate
    return layers.Concatenate()([p1, p2, p3, p4])