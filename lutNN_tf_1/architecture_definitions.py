import numpy as np
###################################################
## OMTF definitions
###################################################
nRefLayers = 8
nLayers = 18
nPDFBins = 2**7
minProbability = 0.001
minPlog = np.log(minProbability)
nPdfValBits = 6
refLayers = [0, 7, 2, 6, 16, 4, 10, 11]
###################################################
## Classic NN definitions
###################################################
# parameters of the classic NN
dense_layer1_size = 16 * 8
dense_layer2_size = 8 * 8
dense_layer3_size = 8 * 6
dense_layer4_size = 1
###################################################
###################################################
def print_Classic_NN():
    print("Classic NN definitions:")
    print("dense_layer1_size:",dense_layer1_size)
    print("dense_layer2_size:",dense_layer2_size)
    print("dense_layer3_size:",dense_layer3_size)
    print("dense_layer4_size:",dense_layer4_size)
    print("------------------------")
###################################################
###################################################
def get_classic_nn_dir_postfix():
    return "_classic_" + str(dense_layer1_size) + "_" + str(dense_layer2_size) + "_" + str(dense_layer3_size)+ "_" + str(dense_layer4_size)
###################################################
## LUT NN definitions
###################################################
# parameters of the LUT NN
oneOverPt = False 
lut_nn = True
output_type = 0
last_input_is_bias = True

input_I = 10
input_F = 4
networkInputSize = nLayers + 1;

layer1_lut_size = 1 << input_I
layer1_neurons = 16 
layer1_lut_I = 3
layer1_lut_F = 10

layer1_output_I = 4
# 4 bits are for the count of the noHit layers which goes to the input of the layer2, other 4 for layer1_output_I
layer2_input_I = layer1_output_I + 4
layer2_lut_size = 1 << layer2_input_I
layer2_neurons = 8  #9 if the charge output is used
layer2_lut_I = 5
layer2_lut_F = 10

layer2_lutRangesCnt = 16
layer2_input_offset = layer2_lut_size / layer2_lutRangesCnt / 2

layer3_input_I = 5
layer3_lut_size = 1 << layer3_input_I
layer3_neurons = 1
layer3_lut_I = 6
layer3_lut_F = 10

# for LUT NN input conversion
rangeFactor = np.full(
  shape = nLayers,
  fill_value = 2 * 2,
  dtype = int
)

rangeFactor[1] = 8*2
rangeFactor[3] = 4*2*2
rangeFactor[5] = 4*2
rangeFactor[9] = 1*2
###################################################
###################################################
def get_lut_nn_dir_postfix():
    return "_lut_" + str(layer1_neurons) + "_" + str(layer2_neurons) + "_" + str(layer3_neurons)
###################################################
###################################################
def print_LUT_NN():
    print("LUT NN definitions:")
    print("layer1_lut_size", layer1_lut_size)
    print("layer2_lut_size", layer2_lut_size)
    print("layer3_lut_size", layer3_lut_size)
    print("layer2_lutRangesCnt", layer2_lutRangesCnt)
    print("layer2_input_offset", layer2_input_offset)
    print("------------------------")
###################################################
###################################################
def custom_loss3(y_true, y_pred):
    loss = y_true[ : , 2] * tf.keras.losses.MAE(y_true[ : , 0], y_pred[ : , 0]) 
    loss += (1 - y_true[ : , 2] ) * tf.keras.losses.MAE(y_true[ : , 1], y_pred[ : , 1]) 
    loss + tf.keras.losses.MSE(y_true[ : , 2], y_pred[ : , 2])
    return loss

###################################################
###################################################
