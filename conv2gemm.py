import layerFuserHelper
def conv2gemm(conv_layer):
    W = conv_layer[0]
    H = conv_layer[1]
    C = conv_layer[2]
    N = conv_layer[3]
    K = conv_layer[4]
    S = conv_layer[5]
    R = conv_layer[6]
    Wpad = conv_layer[7]
    Hpad = conv_layer[8]
    Wstride = conv_layer[9]
    Hstride = conv_layer[10]
    

    (E,F) = layerFuserHelper.infer_output_size(conv_layer)
    gemmM = E * F
    gemmK = R * S * C
    gemmN = K
    return (gemmM, gemmK, gemmN)

