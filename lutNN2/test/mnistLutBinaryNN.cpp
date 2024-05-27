/*
 * mnistLutBinaryNN.cpp
 *
 *  Created on: Feb 6, 2022
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/LutLayer.h"

#include <time.h>
#include <random>
#include <boost/timer/timer.hpp>

using namespace lutNN;
using namespace std;
using namespace boost::timer;

int main(void) {

    LutFloatLayer lutLayer(10, 10, 6);

    LutBinaryLayer lutBinaryLayer(10, 10, 6);

    //run(boost::dynamic_bitset<>& inputs)

    lutBinaryLayer.run(lutBinaryLayer.getLastOutVals());

    lutLayer.run(lutBinaryLayer.getLastOutVals());
}

