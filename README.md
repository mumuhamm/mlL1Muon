## Installation instructions:

```
git clone ssh://git@gitlab.cern.ch:7999/kbunkow/machine_learning.git
cd machine_learning
ln -s ${PWD}/lutNN2 lutNN
ln -s /cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/hls/2019.08-bcolbf hls
mkdir build
cd build
cmake ../
make -j 4 install
```
