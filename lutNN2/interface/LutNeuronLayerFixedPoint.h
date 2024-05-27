//============================================================================
// Name        : LutNeuronLayerFixedPoint.h
// Author      : Karol Bunkowski
// Created on: Mar 12, 2021
// Version     :
// Copyright   : All right reserved
// Description : Fixed point LUT layer
//============================================================================

#ifndef INTERFACE_LUTNEURONLAYERFIXEDPOINT_H_
#define INTERFACE_LUTNEURONLAYERFIXEDPOINT_H_

#include <ap_fixed.h>
#include <ap_int.h>
#include <array>
#include <limits>

#include <boost/property_tree/ptree.hpp>

#include "lutNN/lutNN2/interface/LutLayerFixedPoint.h"
#include "lutNN/lutNN2/interface/Node.h"
#include "LutNetworkFixedPointCommon.h"

namespace lutNN {


template<int input_I, int input_F, std::size_t inputSize,
         int lut_I,   int lut_F,
         int neurons, int output_I>
class LutNeuronLayerFixedPoint {
public:
    static const int input_W = input_I + input_F;
    static const int lut_W   = lut_I   + lut_F;

    //the lut out values sum
    static const int lutOutSum_I  = lut_I + ceil(log2(inputSize));
    static const int lutOutSum_W  = lutOutSum_I + lut_F;

    static const int output_W = output_I + lut_F;

    //static_assert( (1<<input_I) <= lutSize);
    static const std::size_t lutSize = 1<<input_I;

    typedef std::array<ap_ufixed<input_W, input_I, AP_TRN, AP_SAT> , inputSize> inputArrayType;

    typedef std::array<ap_fixed<lutOutSum_W, lutOutSum_I> , neurons> lutSumArrayType;

    LutNeuronLayerFixedPoint()  { //FIXME initialise name(name)
        //static_assert(lut_I <= (output_I - ceil(log2(inputSize)) ), "not correct lut_I, output_I  and inputSize"); //TODO

        std::cout<<"Constructing LutNeuronLayerFixedPoint "<<name
                 <<"\n     input_I  "<<std::setw(2)<<input_I   <<"    input_F "<<std::setw(2)<<input_F<<" input_W "<<std::setw(2)<<input_W<<" inputSize "<<std::setw(2)<<inputSize
                 <<"\n       lut_I  "<<std::setw(2)<<lut_I     <<"      lut_F "<<std::setw(2)<<  lut_F<<"   lut_W "<<std::setw(2)<<  lut_W<<"   lutSize "<<std::setw(2)<<lutSize
                 <<"\n lutOutSum_I "<<std::setw(2)<<lutOutSum_I<<" lutOutSum_W "<<std::setw(2)<<  lutOutSum_W
                 <<"\n    output_I "<<std::setw(2)<<   output_I<<"    output_W "<<std::setw(2)<<  output_W
                 <<"\n neurons "<<std::setw(2)<<neurons
                 <<"\n outOffset "<<outOffset<<" = "<<std::hex<<outOffset<<" width "<<outOffset.width<<std::dec
                 <<std::endl;

        //std::cout<<"(output_I - ceil(log2(inputSize)) ) "<<(output_I - ceil(log2(inputSize)) )<<std::endl;

        //if(lut_I != (output_I - ceil(log2(inputSize)) ) )
        //    throw std::runtime_error("LutLayerFixedPoint: lut_I != (output_I - ceil(log2(inputSize))");
    }

    virtual ~LutNeuronLayerFixedPoint() {

    }

    void setName(std::string name) {
        this->name = name;
    }

    //std::array< std::array<std::array<ap_fixed<output_W, output_I>, lutSize>, neurons>, inputSize>&
    auto& getLutArray() {
        return lutArray;
    }

    void setLutArray(
            const std::array<std::array<std::array<ap_fixed<output_W, output_I>, lutSize>, neurons>, inputSize> &lutArray) {
        this->lutArray = lutArray;
    }

    void initLuts(NodeLayer::iterator layerBegin, NodeLayer::iterator layerEnd);

    void initLuts(NodeLayer& layer) {
        initLuts(layer.begin(), layer.end());
    }

    void save(boost::property_tree::ptree& tree, std::string keyPath) {
        PUT_VAR(tree, keyPath + "." + name, input_I)
        PUT_VAR(tree, keyPath + "." + name, input_F)
        PUT_VAR(tree, keyPath + "." + name, inputSize)
        PUT_VAR(tree, keyPath + "." + name, lut_I)
        PUT_VAR(tree, keyPath + "." + name, lut_F)
        PUT_VAR(tree, keyPath + "." + name, neurons)
        PUT_VAR(tree, keyPath + "." + name, output_I)

        for(unsigned int iInput = 0; iInput < lutArray.size(); iInput++) {
            for(unsigned int iNeuron = 0; iNeuron < lutArray[iInput].size(); iNeuron++) {
                auto& lut = lutArray.at(iInput).at(iNeuron);
                std::ostringstream ostr;
                for(auto& a : lut) {
                    ostr<<std::hex<<a.bits_to_uint64()<<", ";
                }
                tree.put(keyPath + "." + name + ".lutArray." + std::to_string(iInput) + "." + std::to_string(iNeuron), ostr.str());
            }
        }
    }

    void load(boost::property_tree::ptree& tree, std::string keyPath) {
        CHECK_VAR(tree, keyPath + "." + name, input_I)
        CHECK_VAR(tree, keyPath + "." + name, input_F)
        CHECK_VAR(tree, keyPath + "." + name, inputSize)
        CHECK_VAR(tree, keyPath + "." + name, lut_I)
        CHECK_VAR(tree, keyPath + "." + name, lut_F)
        CHECK_VAR(tree, keyPath + "." + name, neurons)
        CHECK_VAR(tree, keyPath + "." + name, output_I)

        for(unsigned int iInput = 0; iInput < lutArray.size(); iInput++) {
            for(unsigned int iNeuron = 0; iNeuron < lutArray[iInput].size(); iNeuron++) {
                auto& lut = lutArray.at(iInput).at(iNeuron);
                auto str = tree.get<std::string>(keyPath + "." + name + ".lutArray." + std::to_string(iInput) + "." + std::to_string(iNeuron));

                std::stringstream ss(str);
                std::string item;

                for(auto& a : lut) {
                    if(std::getline(ss, item, ',') ) {
                        a.setBits(std::stoull(item, NULL, 16));
                    }
                    else {
                        throw std::runtime_error("LutNeuronLayerFixedPoint::read: number of items get from file is smaller than lut size");
                    }
                }
            }
        }
    }

    lutSumArrayType&
    runWithInterpolation(const inputArrayType& inputArray) {
        for(unsigned int iNeuron = 0; iNeuron < lutOutSumArray.size(); iNeuron++) {
            auto& lutOutSum = lutOutSumArray.at(iNeuron);
            lutOutSum = 0;
            for(unsigned int iInput = 0; iInput < inputArray.size(); iInput++) {
                auto address = inputArray.at(iInput).to_uint(); //address in principle is unsigned
                auto& lut = lutArray.at(iInput).at(iNeuron);

                auto addresPlus1 = address +1;
                if(addresPlus1 >= lut.size())
                    addresPlus1 = address;

                auto derivative = lut.at(addresPlus1) - lut.at(address); // must be signed

                //N.B. the address and fractionalPart is the same for all neurons, what matters for the firmware
                ap_ufixed<input_W-input_I, 0> fractionalPart = inputArray.at(iInput);

                auto result = lut.at(address) + fractionalPart * derivative;
                lutOutSum += result;

                /*std::cout<<__FUNCTION__<<":"<<__LINE__<<name<<" "<<" iNeuron "<<std::setw(3)<<iNeuron<<" iInput "<<std::setw(8)<<iInput<<" input "<<std::setw(6)<<inputArray.at(iInput)<<" address "<<std::setw(5)<<address
                         //<<" fractionalPart "<<std::setw(8)<<fractionalPart//<<" width "<<fractionalPart.width<<" iwidth "<<fractionalPart.iwidth
                         //<<" derivative "<<std::setw(8)<<derivative//<<" width "<<derivative.width<<" iwidth "<<derivative.iwidth
                         //<<" lut[addresPlus1] "<<std::setw(8)<<lut.at(addresPlus1)<<" lut[addr] "<<std::setw(8) << lut.at(address)
                         //<<" outVal "<<std::setw(8)<<lutSum<<" outVal-offset "<<(lutSum - 1.77778)<<std::endl;
                         <<" result "<<std::setw(8)<<result<<std::endl;*/

            }

            /*std::cout<<__FUNCTION__<<":"<<__LINE__<<name<<" "<<" iNeuron "<<iNeuron<<" lutOutSum "<<std::setw(10)<<lutOutSum
                     <<" width "<<lutOutSum.width<<" iwidth "<<lutOutSum.iwidth<<std::endl;*/
        }

        return lutOutSumArray;
    }

    //Output without offset
    auto& getLutOutSum() {
        return lutOutSumArray;
    }

    //converts the output values from signed to unsigned by adding the offset = 1 << (output_I-1)
    //these values can be then directly used as inputs of the next LUT layer
    auto& getOutWithOffset() {
        for(unsigned int iOut = 0; iOut < lutOutSumArray.size(); iOut++) {
            outputArray[iOut] = lutOutSumArray[iOut] + outOffset;

            //std::cout<<__FUNCTION__<<":"<<__LINE__<<name<<" "<<"iOut "<<iOut<<" lutOutSumArray[i] "<<lutOutSumArray[iOut]<<" outputArray[i] "<<outputArray[iOut]<<std::endl;
        }

        return outputArray;
    }

    auto getName() {
        return name;
    }

private:
    lutSumArrayType lutOutSumArray;
    std::array<ap_ufixed<output_W, output_I, AP_TRN, AP_SAT> , neurons> outputArray;

    ap_uint<output_I> outOffset = 1 << (output_I-1);


    std::array< std::array<std::array<ap_fixed<lut_W, lut_I> , lutSize>, neurons>, inputSize> lutArray; //[inputNum][outputNum =  neuronNum][address]

    std::string name;
};


template<int input_I, int input_F, std::size_t inputSize,
         int lut_I,   int lut_F,
         int neurons, int output_I>
void LutNeuronLayerFixedPoint<input_I, input_F, inputSize, lut_I,   lut_F, neurons, output_I>::initLuts(NodeLayer::iterator layerBegin, NodeLayer::iterator layerEnd) {
    unsigned int layerSize = layerEnd - layerBegin;
    if(lutArray.size() * lutArray[0].size() != layerSize) {
        std::cout<<"LutNeuronLayerFixedPoint::initLuts "<<name<<" "<<" lutArray.size() x lutArray[0].size() "<<lutArray.size()<<" x "<<lutArray[0].size()
                <<" = "<<lutArray.size() * lutArray[0].size()
                <<", layer.size() "<<layerSize<<" - not good !!!!!!!!!!"<<std::endl;
        exit(1);
    }

    float maxAPFixedLutValue = max_ap_fixed<lut_W, lut_I>().to_float();

    double maxLutValue = -99999999;
    double minLutValue =  99999999;

    for(auto nodeIt = layerBegin; nodeIt < layerEnd; nodeIt++) {
        auto lutNode = static_cast<LutNode*>(nodeIt->get() );
        for(unsigned int iAddr = 0; iAddr < lutNode->getFloatValues().size(); iAddr++) {
            if(maxLutValue < lutNode->getFloatValues()[iAddr])
                maxLutValue = lutNode->getFloatValues()[iAddr];

            if(minLutValue > lutNode->getFloatValues()[iAddr])
                minLutValue = lutNode->getFloatValues()[iAddr];
        }
    }


    std::cout<<name<<" "<<"LutNeuronLayerFixedPoint::initLuts maxLutValue "<<maxLutValue<<" minLutValue "<<minLutValue<<std::endl;
    std::cout<<" maxFixedLutValue "<<maxAPFixedLutValue<<std::endl;


    for(unsigned int iNeuron = 0; iNeuron < lutArray[0].size(); iNeuron++) {
        for(unsigned int iInput = 0; iInput < lutArray.size(); iInput++) {
            unsigned int iNode = iInput + iNeuron * lutArray.size();
            //std::cout<<name<<" "<<" iNeuron "<<iNeuron<<" iInput "<<std::setw(2)<<iInput<<" iNode "<<iNode<<std::endl;

            auto lutNode = static_cast<LutNode*>( (layerBegin + iNode)->get() );

            if(lutArray.at(iInput).at(iNeuron).size() != lutNode->getFloatValues().size()) {
                std::cout<<"LutNeuronLayerFixedPoint::initLuts "<<name<<" lut size in lutArray"<<lutArray.at(iInput).at(iNeuron).size()
                         <<" lutNode->getFloatValues().size() "<<lutNode->getFloatValues().size()<<" - not good !!!"<<std::endl;
                exit(1);
            }

            auto& lut = lutArray.at(iInput).at(iNeuron);
            for(unsigned int iAddr = 0; iAddr < lut.size(); iAddr++) {
                float value = lutNode->getFloatValues().at(iAddr);

                lut[iAddr] = ap_fixed<lut_W, lut_I,  AP_TRN, AP_SAT>(value);

                std::cout<<" iNeuron "<<iNeuron<<" iInput "<<std::setw(2)<<iInput<<" iNode "<<std::setw(2)<< iNode<<" iAddr "<<std::setw(4)<<iAddr<<" value "<<std::setw(10)<<value<<" lut[iAddr]  "<<lut[iAddr] <<std::endl;

                if(value < -maxAPFixedLutValue ) {
                    //std::cout<<" iNeuron "<<iNeuron<<" iInput "<<std::setw(2)<<iInput<<" iNode "<<std::setw(2)<< iNode<<" iAddr "<<std::setw(4)<<iAddr<<" value "<<std::setw(10)<<value<<" !!!!!!!!!!!!!!!!!!"<<std::endl;
                }
                else if(value > maxAPFixedLutValue) {
                    //std::cout<<" iNeuron "<<iNeuron<<" iInput "<<std::setw(2)<<iInput<<" iNode "<<std::setw(2)<< iNode<<" iAddr "<<std::setw(4)<<iAddr<<" value "<<std::setw(10)<<value<<" maxFixedLutValue "<<maxFixedLutValue<<" !!!!!!!!!!!!!!!!!!"<<std::endl;
                }
            }
            //lut[lut.size() -1] = 0; ////FIXME for the moment ignoring the last address, because in the layer 1 there is noHit value
        }
    }
}

} /* namespace lutNN */

#endif /* INTERFACE_LUTNEURONLAYERFIXEDPOINT_H_ */
