//============================================================================
// Name        : lutNetworkFixedPointTest.cpp
// Author      : Karol Bunkowski
// Created on: Mar 12, 2021
// Version     :
// Copyright   : All right reserved
// Description : lutNN trainer for the omtf
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <random>
#include <boost/timer/timer.hpp>
#include "lutNN/lutNN2/interface/LutNetworkBase.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <lutNN/lutNN2/interface/LutNeuronLayerFixedPoint.h>

#include "TCanvas.h"
#include "TFile.h"
#include "TH2F.h"

#include "lutNN/lutNN2/interface/LutInterNetwork.h"
#include "lutNN/lutNN2/interface/NetworkBuilder.h"
#include "lutNN/lutNN2/interface/LutInter.h"
#include "lutNN/lutNN2/interface/NetworkOutNode.h"
#include "lutNN/lutNN2/interface/EventsGeneratorOmtf.h"
#include "lutNN/lutNN2/interface/Utlis.h"
#include "lutNN/lutNN2/interface/NetworkSerialization.h"
#include "lutNN/lutNN2/interface/OmtfAnalyzer.h"

#include "lutNN/lutNN2/interface/LutNetworkFixedPoint.h"

using namespace lutNN;
using namespace std;
using namespace boost::timer;

int main(void) {
    puts("Hello World!!!");

    const int input_W = 15;
    const int input_I = 10;
    const std::size_t inputSize = 4;
    const int output_W = 15;
    const int output_I = 10;
    const std::size_t outputSize = 2;
    const int lut_W = 15;
    const int lut_I = 7;
    const std::size_t lutSize = 8;

    ap_ufixed<input_W-input_I, 0> test = 0.7;
    cout<<" width "<<test.width<<" iwidth "<<test.iwidth<<endl;

    cout<<"test "<<test<<" "<<test.to_string(2)<<" "<<test.to_string(2, true)<<endl;
    test = 0.07;
    cout<<"test "<<test<<" "<<test.to_string(2)<<" "<<test.to_string(2, true)<<endl;
    test = 10.7;
    cout<<"test "<<test<<" "<<test.to_string(2)<<" "<<test.to_string(2, true)<<endl;

    ap_ufixed<input_W, input_I> test1 = 1023.9;
    cout<<" width "<<test1.width<<" iwidth "<<test1.iwidth<<endl;
    cout<<"test1 "<<test1<<" "<<test1.to_string(2)<<" "<<test1.to_string(2, true)<<endl;

    ap_ufixed<input_W, input_I> test3 = 0;
    cout<<" width "<<test3.width<<" iwidth "<<test3.iwidth<<endl;
    cout<<"test3 "<<test3<<" "<<test3.to_string(2)<<" "<<test3.to_string(2, true)<<endl;

    ap_fixed<input_W+1, input_I+1> test2 = test3-test1;
    cout<<" width "<<test2.width<<" iwidth "<<test2.iwidth<<endl;
    cout<<"test2 "<<test2<<" "<<test2.to_string(2)<<" "<<test2.to_string(2, true)<<endl;

    auto maxVal = max_ap_ufixed<input_W, input_I>();
    cout<<"maxVal "<<maxVal<<" "<<maxVal.to_string(2)<<" width "<<maxVal.width<<" iwidth "<<maxVal.iwidth<<endl;

    //auto maxVal2 = max_ap_ufixed<66, 64>();

    const int layer1_input_I = 10;
    const int layer1_input_F = 3;
    const int layer1_inputSize = 6;
    const int layer1_lut_I = 4;
    const int layer1_lut_F = 4;
    const int layer1_neurons = 8;
    const int layer1_output_I = 6;
    LutNeuronLayerFixedPoint<layer1_input_I, layer1_input_F, layer1_inputSize, layer1_lut_I, layer1_lut_F, layer1_neurons, layer1_output_I> lutLayer;

    auto&  lutArray = lutLayer.getLutArray();//[inputNum][outputNum][address]
    for(unsigned int inputNum = 0; inputNum < lutArray.size(); inputNum++) {
        for(unsigned int outputNum = 0; outputNum < lutArray[inputNum].size(); outputNum++) {
            for(unsigned int address = 0; address < lutArray[inputNum][outputNum].size(); address++) {
                lutArray[inputNum][outputNum][address] = 0;
            }
        }
    }

    //[inputNum][outputNum][address]
    lutArray[0][0][0] = 0;
    lutArray[0][0][1] = 1023.99;
    lutArray[1][0][2] = 1000.;

    lutArray[0][1][0] = 1023;
    lutArray[0][1][1] = 0;


/*    std::array<ap_ufixed<layer1_input_I + layer1_input_F, layer1_input_I> , layer1_inputSize, AP_RND, AP_SAT> inputArray = {0.5, 1.6, 2.7, 3.8};

    cout<<"input";
    for(auto& input : inputArray)
        cout<<" "<<input;

    cout<<endl;

    lutLayer.runWithInterpolation(inputArray);*/

/*
    const int input_intBits   = 4;
    const int input_fractBits = 2;
    const std::size_t networkInputSize = 9;

    const int layer1_neurons = 2;
    const int layer1_output_intBits = 10;
    const int layer1_lut_fractBits = 3;

    const int layer2_neurons = 3;
    const int layer2_output_intBits = 10;
    const int layer2_lut_fractBits = 3;

    const int layer3_neurons = 3;
    const int layer3_output_intBits = 10;
    const int layer3_lut_fractBits = 3;

    const int network_output_W =4;
    const int network_output_I = 4;
    const std::size_t network_outputSize = 1;

    LutNetworkFixedPoint<input_intBits,  input_fractBits,  networkInputSize,
                         layer1_neurons, layer1_output_intBits, layer1_lut_fractBits,        //layer1_lutSize = 2 ^ input_I
                         layer2_neurons, layer2_output_intBits, layer2_lut_fractBits,
                         layer3_neurons, layer3_output_intBits, layer3_lut_fractBits> lutNetwork; //, network_output_W, network_output_I, network_outputSize>


    lutNetwork.runWithInterpolation();
*/
}

