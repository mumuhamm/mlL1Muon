/*
 * LutNetworkFixedPointRegression2Outputs.h
 *
 *  Created on: April 13, 2021
 *      Author: Karol Bunkowski, kbunkow@cern.ch
 */

#ifndef INTERFACE_LUTNETWORKFIXEDPOINTREGRESSION2OUTPUTS_H_
#define INTERFACE_LUTNETWORKFIXEDPOINTREGRESSION2OUTPUTS_H_

#include "lutNN/lutNN2/interface/LutNeuronLayerFixedPoint.h"
//#include "lutNN/lutNN2/interface/LutLayerFixedPoint.h"

#include "ap_int.h"

#include "lutNN/lutNN2/interface/LutNetworkBase.h"
#include <cmath>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace lutNN {


//_I - number of integer bits in the ap_ufixed, _F - number of fractional bits in the ap_ufixed
//the network has two outputs, and since each output can have different range, the LUTs in the last layer have different I and F
template<int input_I,  int input_F,  std::size_t inputSize,
         int layer1_lut_I, int layer1_lut_F, int layer1_neurons,
         int layer1_output_I, //to the layer1 output the bias is added to make the layer1 input
         int layer2_input_I,
         int layer2_lut_I, int layer2_lut_F, int layer2_neurons,
         int layer3_input_I,
         int layer3_0_inputCnt, int layer3_0_lut_I, int layer3_0_lut_F, int output0_I, int output0_F,
         int layer3_1_inputCnt, int layer3_1_lut_I, int layer3_1_lut_F, int output1_I, int output1_F>
class LutNetworkFixedPointRegression2Outputs: public LutNetworkFixedPointRegressionBase {
public:
    LutNetworkFixedPointRegression2Outputs() {
        static_assert(layer2_neurons == (layer3_0_inputCnt + layer3_1_inputCnt));

        std::cout<<"LutNetworkFixedPoint"<<std::endl;
        lutLayer1.setName("lutLayer1");
        lutLayer2.setName("lutLayer2");
        lutLayer3_0.setName("lutLayer3_0");
        lutLayer3_1.setName("lutLayer3_1");
    };

    virtual ~LutNetworkFixedPointRegression2Outputs() {};


    typedef LutNeuronLayerFixedPoint<input_I, input_F, inputSize, layer1_lut_I, layer1_lut_F, layer1_neurons, layer1_output_I> LutLayer1;
    LutLayer1 lutLayer1;

    static const unsigned int noHitCntShift = layer1_output_I; //FIXME should be layer1_output_I ???

    static const int layer2_input_F = layer1_lut_F;

    typedef LutNeuronLayerFixedPoint<layer2_input_I, layer2_input_F, layer1_neurons, layer2_lut_I, layer2_lut_F, layer2_neurons, layer3_input_I> LutLayer2;
    LutLayer2 lutLayer2;

    static const int layer3_input_F = layer2_lut_F;

    typedef LutNeuronLayerFixedPoint<layer3_input_I, layer3_input_F, layer3_0_inputCnt, layer3_0_lut_I, layer3_0_lut_F, 1, output0_I> LutLayer3_0;
    LutLayer3_0 lutLayer3_0; //"lutLayer3_0"

    typedef LutNeuronLayerFixedPoint<layer3_input_I, layer3_input_F, layer3_1_inputCnt, layer3_1_lut_I, layer3_1_lut_F, 1, output1_I> LutLayer3_1;
    LutLayer3_1 lutLayer3_1; //"lutLayer3_1"

    void runWithInterpolation() {
        lutLayer1.runWithInterpolation(inputArray);
        auto& layer1Out = lutLayer1.getOutWithOffset();

        std::array<ap_ufixed<layer2_input_I + layer2_input_F, layer2_input_I, AP_TRN, AP_SAT> , layer1_neurons> layer1OutWithBias;
        for(unsigned int i = 0; i < layer1Out.size(); i++) {
            layer1OutWithBias[i] = layer1Out[i] + layer1Bias;
            //std::cout<<"i "<<i<<" layer1Out[i] "<<layer1Out[i]<<" layer1OutWithBias "<<i<<" "<<layer1OutWithBias[i]<<" layer1Bias "<<layer1Bias<<std::dec<<std::endl;
        }

        lutLayer2.runWithInterpolation(layer1OutWithBias);
        auto& layer2Out = lutLayer2.getOutWithOffset();

        typename LutLayer3_0::inputArrayType lutLayer3_0_input;
        std::copy(layer2Out.begin(), layer2Out.begin() + lutLayer3_0_input.size(), lutLayer3_0_input.begin());

        typename LutLayer3_1::inputArrayType lutLayer3_1_input;
        std::copy(layer2Out.begin() + lutLayer3_0_input.size(), layer2Out.end(), lutLayer3_1_input.begin());

        lutLayer3_0.runWithInterpolation(lutLayer3_0_input);
        lutLayer3_1.runWithInterpolation(lutLayer3_1_input);
    }


    void run(std::vector<float>& inputs, float noHitVal, std::vector<double>& nnResult) override {
        unsigned int noHitsCnt = 0;
        //here the last input is an additional input with noHitsCnt, below we just crosscheck it
        for(unsigned int iInput = 0; iInput < inputs.size() -1; iInput++) {
            inputArray[iInput] = inputs[iInput];
            if(inputs[iInput] == noHitVal)
                noHitsCnt++;
        }

        unsigned int bias = (noHitsCnt << noHitCntShift);
        if(noHitsCnt != inputs.back()) {
            std::cout<<"LutNetworkFixedPointRegression2Outputs::run(): noHitsCnt != inputs.back(), noHitsCnt "<<noHitsCnt<<" inputs.back() "<<inputs.back() <<std::endl;
            for(unsigned int iInput = 0; iInput < inputArray.size(); iInput++) {
                std::cout<<"iInput "<<iInput<<" val "<<inputArray[iInput]<<std::endl;
            }
            return;
        }
        //std::cout<<" noHitsCnt "<<noHitsCnt<<" event->noHitVal "<<event->noHitVal<<" bias "<<std::hex<<"0x"<<bias<<std::dec<<std::endl;

        //layer1Bias switches the input of the layer2 (i.e. output of the layer1) do different regions in the LUTs
        //depending on the  number of layers without hits
        layer1Bias = bias;
        //std::cout<<"layer1Bias "<<layer1Bias<<" W "<<layer1Bias.width <<std::endl;

        runWithInterpolation();

        //output0_I goes to the declaration of the lutLayer3_0, but it does not matter, as it is used only for the outputArray
        auto layer3_0_out = ap_ufixed<output0_I+output0_F, output0_I, AP_RND_CONV, AP_SAT>(lutLayer3_0.getLutOutSum()[0]); //TODO should be AP_RND_CONV rather, but it affect the rate
        auto layer3_1_out = ap_fixed <output1_I+output1_F, output1_I, AP_RND_CONV, AP_SAT>(lutLayer3_1.getLutOutSum()[0]); //here layer3_0_out has size 1
        //auto layer3_0_out = lutLayer3_0.getLutOutSum()[0]; //here layer3_0_out has size 1
        //auto layer3_1_out = lutLayer3_1.getLutOutSum()[0]; //here layer3_0_out has size 1

        //std::cout<<"layer3_0_out[0] "<<layer3_0_out[0]<<" layer3_1_out[0] "<<layer3_1_out[0]<<std::endl;

        nnResult[0] = layer3_0_out.to_float();
        nnResult[1] = layer3_1_out.to_float();
    }

    //pt in the hardware scale, ptGeV = (ptHw -1) / 2
    int getCalibratedHwPt() override {
        auto lutAddr = ap_ufixed<output0_I+output0_F+output0_F, output0_I+output0_F, AP_RND_CONV, AP_SAT>(lutLayer3_0.getLutOutSum()[0]);
        lutAddr = lutAddr<<output0_F;
        //std::cout<<"lutLayer3_0.getLutOutSum()[0] "<<lutLayer3_0.getLutOutSum()[0]<<" lutAddr.to_uint() "<<lutAddr.to_uint()<<" ptCalibrationArray[lutAddr] "<<ptCalibrationArray[lutAddr.to_uint()]<<std::endl;
        return ptCalibrationArray[lutAddr.to_uint()].to_uint();
    }

    void initLuts(LutNetworkBase& network) {
        auto& layers = network.getLayers();

        std::cout<<"layers.at(0).size() "<<layers.at(0).size()<<std::endl;
        lutLayer1.initLuts(layers.at(0));

        std::cout<<"layers.at(2).size() "<<layers.at(2).size()<<std::endl;
        lutLayer2.initLuts(layers.at(2));

        std::cout<<"layers.at(4).size() "<<layers.at(4).size()<<std::endl;

        lutLayer3_0.initLuts(layers.at(4).begin(), layers.at(4).begin() + layer3_0_inputCnt);

        lutLayer3_1.initLuts(layers.at(4).begin() + layer3_0_inputCnt, layers.at(4).end());
    }

    void save(const std::string &filename) override {
        // Create an empty property tree object.
        boost::property_tree::ptree tree;

        lutLayer1.save(tree, "LutNetworkFixedPointRegression2Outputs");
        lutLayer2.save(tree, "LutNetworkFixedPointRegression2Outputs");
        lutLayer3_0.save(tree, "LutNetworkFixedPointRegression2Outputs");
        lutLayer3_1.save(tree, "LutNetworkFixedPointRegression2Outputs");

        int size = ptCalibrationArray.size();
        std::string key = "LutNetworkFixedPointRegression2Outputs.ptCalibrationArray";
        PUT_VAR(tree, key, size)
        std::ostringstream ostr;
        for(auto& a : ptCalibrationArray) {
            //ostr<<std::hex<<a.to_uint()<<", ";
            ostr<<a.to_uint()<<", ";
        }
        tree.put(key + ".values", ostr.str());

        boost::property_tree::write_xml(filename, tree, std::locale(), boost::property_tree::xml_parser::xml_writer_make_settings<std::string>(' ', 2));
    }

    void load(const std::string &filename) override {
        // Create an empty property tree object.
        boost::property_tree::ptree tree;

        boost::property_tree::read_xml(filename, tree);

        lutLayer1.load(tree, "LutNetworkFixedPointRegression2Outputs");
        lutLayer2.load(tree, "LutNetworkFixedPointRegression2Outputs");
        lutLayer3_0.load(tree, "LutNetworkFixedPointRegression2Outputs");
        lutLayer3_1.load(tree, "LutNetworkFixedPointRegression2Outputs");


        std::string key = "LutNetworkFixedPointRegression2Outputs.ptCalibrationArray";
        int size = ptCalibrationArray.size();
        CHECK_VAR(tree, key, size)

        auto str = tree.get<std::string>(key + ".values");

        std::stringstream ss(str);
        std::string item;

        for(auto& a : ptCalibrationArray) {
            if(std::getline(ss, item, ',') ) {
                //a = std::stoul(item, NULL, 16);
                a = std::stoul(item, NULL, 10);
            }
            else {
                throw std::runtime_error("LutNetworkFixedPointRegression2Outputs::read: number of items get from file is smaller than lut size");
            }
        }
    }

    auto& getPtCalibrationArray() {
        return ptCalibrationArray;
    }

private:
    std::array<ap_ufixed<LutLayer1::input_W, input_I, AP_TRN, AP_SAT> , inputSize> inputArray;
    ap_uint<layer2_input_I> layer1Bias;

    //ptCalibrationArray size should be 1024, the LSB of the input 0.25 GeV,
    //the output is int, with range 0...511, the LSB of output 0.5 GeV
    std::array<ap_uint<9>, 1<<(output0_I+output0_F)> ptCalibrationArray;
};

} /* namespace lutNN */

#endif /* INTERFACE_LUTNETWORKFIXEDPOINTREGRESSION2OUTPUTS_H_ */
