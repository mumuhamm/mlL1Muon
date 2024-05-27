/*
 * LutNetworkFixedPoint.h
 *
 *  Created on: Mar 16, 2021
 *      Author: kbunkow
 */

#ifndef INTERFACE_LUTNETWORKFIXEDPOINT_H_
#define INTERFACE_LUTNETWORKFIXEDPOINT_H_

#include "lutNN/lutNN2/interface/LutNeuronLayerFixedPoint.h"
#include "lutNN/lutNN2/interface/LutLayerFixedPoint.h"

#include "lutNN/lutNN2/interface/Event.h"
#include "lutNN/lutNN2/interface/NetworkOutNode.h"
#include "lutNN/lutNN2/interface/LutNetworkBase.h"
#include <cmath>

namespace lutNN {

//_I - number of integer bits in the ap_ufixed, _F - number of fractional bits in the ap_ufixed
template<int input_I,  int input_F,  std::size_t inputSize,
         int layer1_lut_I, int layer1_lut_F, int layer1_neurons,
         int layer1_output_I, //too the layer1 output the bias is added to make the layer1 input
         int layer2_input_I,
         int layer2_lut_I, int layer2_lut_F, int layer2_neurons,
         int layer3_input_I,
         int layer3_lut_I, int layer3_lut_F, int layer3_neurons>//,
         //int output_W, int output_I, std::size_t outputSize>
class LutNetworkFixedPoint {
public:
    LutNetworkFixedPoint(): outputNode(nullptr) {
        std::cout<<"LutNetworkFixedPoint"<<std::endl;
        lutLayer1.setName("lutLayer1");
        lutLayer2.setName("lutLayer2");
        lutLayer3.setName("lutLayer3");
    };

    LutNetworkFixedPoint(NetworkOutNode* outputNode): outputNode(outputNode) {
        std::cout<<"LutNetworkFixedPoint"<<std::endl;
    };
    virtual ~LutNetworkFixedPoint() {};


    typedef LutNeuronLayerFixedPoint<input_I, input_F, inputSize, layer1_lut_I, layer1_lut_F, layer1_neurons, layer1_output_I> LutLayer1;
    LutLayer1 lutLayer1;

    static const unsigned int noHitCntShift = layer1_output_I;

    static const int layer2_input_F = layer1_lut_F;

    typedef LutNeuronLayerFixedPoint<layer2_input_I, layer2_input_F, layer1_neurons, layer2_lut_I, layer2_lut_F, layer2_neurons, layer3_input_I> LutLayer2;
    LutLayer2 lutLayer2;

    static const int layer3_input_F = layer2_lut_F;

    typedef LutLayerFixedPoint<layer3_input_I, layer3_input_F, layer2_neurons, layer3_lut_I, layer3_lut_F> LutLayer3;
    LutLayer3 lutLayer3; //"lutLayer3"

    auto& runWithInterpolation() {
        lutLayer1.runWithInterpolation(inputArray);
        auto& layer1Out = lutLayer1.getOutWithOffset();

        std::array<ap_ufixed<layer2_input_I + layer2_input_F, layer2_input_I, AP_TRN, AP_SAT> , layer1_neurons> layer1OutWithBias;
        for(unsigned int i = 0; i < layer1Out.size(); i++) {
            layer1OutWithBias[i] = layer1Out[i] + layer1Bias;
            //std::cout<<"i "<<i<<" layer1Out[i] "<<layer1Out[i]<<" layer1OutWithBias "<<i<<" "<<layer1OutWithBias[i]<<" layer1Bias "<<layer1Bias<<std::dec<<std::endl;
        }

        lutLayer2.runWithInterpolation(layer1OutWithBias);
        auto& layer2Out = lutLayer2.getOutWithOffset();
        auto& layer3Out = lutLayer3.runWithInterpolation(layer2Out);

        return layer3Out;
    }


    template <typename InputType>
    void run(Event<InputType>* event) {
        unsigned int noHitsCnt = 0;
        for(unsigned int iInput = 0; iInput < event->inputs.size(); iInput++) {
            inputArray[iInput] = event->inputs[iInput];
            if(event->inputs[iInput] == event->noHitVal)
                noHitsCnt++;
        }

        unsigned int bias = (noHitsCnt << noHitCntShift);
        //std::cout<<" noHitsCnt "<<noHitsCnt<<" event->noHitVal "<<event->noHitVal<<" bias "<<std::hex<<"0x"<<bias<<std::dec<<std::endl;

        //layer1Bias switches the input of the layer2 (i.e. output of the layer1) do different regions in the LUTs
        //depending on the  number of layers without hits
        layer1Bias = bias;
        //std::cout<<"layer1Bias "<<layer1Bias<<" W "<<layer1Bias.width <<std::endl;

        auto& outputFixedPoint = runWithInterpolation();

        std::vector<double> outputDouble(outputFixedPoint.size());

        for(unsigned int iOut = 0; iOut < outputFixedPoint.size(); iOut++) {
            outputDouble[iOut] = outputFixedPoint[iOut];
        }

        double maxInVal = 0;
        SoftMax::softMax(outputDouble, maxInVal, event->nnResult);
    }


    void initLuts(LutNetworkBase& network) {
        auto& layers = network.getLayers();

        std::cout<<"layers.at(0).size() "<<layers.at(0).size()<<std::endl;
        lutLayer1.initLuts(layers.at(0));

        std::cout<<"layers.at(2).size() "<<layers.at(2).size()<<std::endl;
        lutLayer2.initLuts(layers.at(2));

        std::cout<<"layers.at(4).size() "<<layers.at(4).size()<<std::endl;
        lutLayer3.initLuts(layers.at(4));
    }

private:
    std::array<ap_ufixed<LutLayer1::input_W, input_I, AP_TRN, AP_SAT> , inputSize> inputArray;
    ap_uint<layer2_input_I> layer1Bias;

    std::unique_ptr<NetworkOutNode> outputNode;
};

} /* namespace lutNN */

#endif /* INTERFACE_LUTNETWORKFIXEDPOINT_H_ */
