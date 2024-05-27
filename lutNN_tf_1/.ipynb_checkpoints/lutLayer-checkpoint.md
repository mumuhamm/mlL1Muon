**Without batches**

address bits = 4

```
input tf.Tensor([0 0 1 0 1 0 0 1 0 1 1 1 0 1 1 0], shape=(16,), dtype=int32)
```
*input of get_lut_values:*

```
input_shaped tf.Tensor(
[[0 0 1 0] <<bits of one address
 [1 0 0 1]
 [0 1 1 1]
 [0 1 1 0]], shape=(4, 4), dtype=int32)


get_lut_values 
addresses: tf.Tensor(
[[ 4]
 [ 9]
 [14]
 [ 6]], shape=(4, 1), dtype=int32)
 
binary_lut_layer 
luts_int: <tf.Variable 'luts_int:0' shape=(4, 16) dtype=int8, numpy=
array([[1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], <<<<<<<<<<one neuron
       [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
       [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
       [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]], dtype=int8)> 
       
y = tf.gather(luts_int, addresses, axis = 1, batch_dims=1)

y - lutValues: tf.Tensor(
[[1]
 [1]
 [0]
 [0]], shape=(4, 1), dtype=int8)
       
```