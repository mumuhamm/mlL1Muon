## Running instructions:

### LUT neural network:

Jupyter version:

* check if the data is read correctly:
    [IO_tests.ipynb](lutNN_tf_1/IO_tests.ipynb)
    
* training: 
    [LUN_NN_training.ipynb](lutNN_tf_1/LUN_NN_training.ipynb)
   
* pereormance plots:
    [LUT_NN_performance.ipynb](lutNN_tf_1/LUT_NN_performance.ipynb) 
   
Shell version:   

```Shell
python3 LUN_NN_training.py
python3 LUT_NN_performance.py
```

### Classic neural network:

Jupyter version:

* check if the data is read correctly:
    [IO_tests.ipynb](lutNN_tf_1/IO_tests.ipynb)
    
* training: 
    [Classic_NN_training.ipynb](lutNN_tf_1/Classic_NN_training.ipynb)
   
* pereormance plots:
    [Classic_NN_performance.ipynb](lutNN_tf_1/Classic_NN_performance.ipynb) 
   
Shell version:   

```Shell
python3 Classic_NN_training.py
python3 Classic_NN_performance.py
```
   
   

* [LUN_NN_training.ipynb](lutNN_tf_1/LUN_NN_training.ipynb)
[LUT_NN_performance.ipynb](lutNN_tf_1/LUT_NN_performance.ipynb)

## Code organization

### Notebooks:

* [IO_tests.ipynb](lutNN_tf_1/IO_tests.ipynb) - notebook with checks of the IO performance

* [LUN_NN_training.ipynb](lutNN_tf_1/LUN_NN_training.ipynb) - notebook with cells for running the LUT NN training

* [LUT_NN_performance.ipynb](lutNN_tf_1/LUT_NN_performance.ipynb) - notebook with cells for making the performance plots using a LUT model loaded from disk

* [Classic_NN_training.ipynb](lutNN_tf_1/Classic_NN_training.ipynb) - notebook with cells for running the Classic NN training

* [Classic_NN_performance.ipynb](lutNN_tf_1/Classic_NN_performance.ipynb) - notebook with cells for making the performance plots using a Classic model loaded from disk

### Functions sets:

* [architecture_definitions.py](lutNN_tf_1/architecture_definitions.py) - definitions of the classic and LUT based NNs

* [model_functions.py](lutNN_tf_1/model_functions.py) - functions returning models

* [io_functions.py](lutNN_tf_1/io_functions.py) - functions related to reading and preprocessing data

* [utility_functions.py](lutNN_tf_1/utility_functions.py) - functions related to saving model results to Pandas DataFrame

* [plotting_functions.py](lutNN_tf_1/plotting_functions.py) - functions form making any kind of plots