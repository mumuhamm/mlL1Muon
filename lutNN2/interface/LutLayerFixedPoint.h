/*
 * LutLayerFixedPoint.h
 *
 *  Created on: Mar 18, 2021
 *      Author: kbunkow
 */

#ifndef INTERFACE_LUTLAYERFIXEDPOINT_H_
#define INTERFACE_LUTLAYERFIXEDPOINT_H_

#include <ap_fixed.h>
#include <ap_int.h>
#include <array>
#include <limits>


#include "lutNN/lutNN2/interface/LutNode.h"
#include "LutNetworkFixedPointCommon.h"

namespace lutNN {

template<int input_I, int input_F, std::size_t inputSize,
         int lut_I,   int lut_F>
class LutLayerFixedPoint {
public:
    static const int input_W = input_I + input_F;
    static const int lut_W   = lut_I   + lut_F;

    static const int output_I = lut_I + ceil(log2(inputSize));

    static const int output_W = output_I + lut_F;

    //static_assert( (1<<input_I) <= lutSize);
    static const std::size_t lutSize = 1<<input_I;

    LutLayerFixedPoint() {//std::string name : name(name)
        std::cout<<"Constructing LutLayerFixedPoint "<<name
                 <<"\n     input_I  "<<std::setw(2)<<input_I   <<"    input_F "<<std::setw(2)<<input_F<<" input_W "<<std::setw(2)<<input_W<<" inputSize "<<std::setw(2)<<inputSize
                 <<"\n       lut_I  "<<std::setw(2)<<lut_I     <<"      lut_F "<<std::setw(2)<<  lut_F<<"   lut_W "<<std::setw(2)<<  lut_W<<"   lutSize "<<std::setw(2)<<lutSize
                 <<"\n    output_I "<<std::setw(2)<<   output_I<<"    output_W "<<std::setw(2)<<  output_W
                 <<std::endl;
    };

    virtual ~LutLayerFixedPoint() {};

    void setName(std::string name) {
        this->name = name;
    }

    std::array<ap_fixed<output_W, output_I> , inputSize>&
    runWithInterpolation(const std::array<ap_ufixed<input_W, input_I, AP_RND, AP_SAT> , inputSize>& inputArray) {
        for(unsigned int iInput = 0; iInput < inputArray.size(); iInput++) {
            auto address = inputArray.at(iInput).to_uint(); //address in principle is unsigned
            auto& lut = lutArray.at(iInput);

            //ap_fixed<output_W+1, output_I>
            auto addresPlus1 = address +1;
            if(addresPlus1 >= lut.size())
                addresPlus1 = address;

            auto derivative = lut.at(addresPlus1) - lut.at(address); // must be signed

            ap_ufixed<input_W-input_I, 0> fractionalPart = inputArray.at(iInput);

            outputArray.at(iInput) = lut.at(address) + fractionalPart * derivative;

            /*std::cout<<__FUNCTION__<<":"<<__LINE__<<name<<" "<<" iInput "<<iInput<<" input "<<std::setw(10)<<inputArray.at(iInput)<<" address "<<address
                    //<<" fractionalPart "<<std::setw(10)<<fractionalPart<<" width "<<fractionalPart.width<<" iwidth "<<fractionalPart.iwidth
                    //<<" derivative "<<std::setw(10)<<derivative<<" width "<<derivative.width<<" iwidth "<<derivative.iwidth
                    //<<" lut[addresPlus1] "<<std::setw(10)<<lut.at(addresPlus1)<<" lut[addr] "<<std::setw(10) << lut.at(address)
                    <<" outVal "<<std::setw(10)<<outputArray.at(iInput)<<std::endl;*/
        }

        return outputArray;
    }

    void initLuts(NodeLayer& layer);

private:
    std::array<ap_fixed<output_W, output_I> , inputSize> outputArray;

    std::array<std::array<ap_fixed<lut_W, lut_I> , lutSize>, inputSize> lutArray; //[inputNum][address]

    std::string name;
};


template<int input_I, int input_F, std::size_t inputSize,
         int lut_I,   int lut_F>
void LutLayerFixedPoint<input_I, input_F, inputSize, lut_I, lut_F>::initLuts(NodeLayer& layer) {
    if(lutArray.size() != layer.size()) {
        std::cout<<"LutLayerFixedPoint::initLuts "<<name<<" "<<" lutArray.size() "<<lutArray.size()<<" layer.size() "<<layer.size()<<" - not good !!!!!!!!!!"<<std::endl;
        exit(1);
    }

    float maxFixedLutValue = max_ap_fixed<lut_W, lut_I>().to_float(); //TODO do it for ap_fixed

    double maxLutValue = -99999999;
    for(unsigned int iNode = 0; iNode < layer.size(); iNode++) {
        auto lutNode = static_cast<LutNode*>(layer[iNode].get() );
        for(unsigned int iAddr = 0; iAddr < lutNode->getFloatValues().size() -1; iAddr++) { //FIXME for the moment ignoring the last addres, because in the layer 1 there is noHit value
            if(maxLutValue < lutNode->getFloatValues()[iAddr])
                maxLutValue = lutNode->getFloatValues()[iAddr];
        }
    }

    std::cout<<name<<" LutLayerFixedPoint::initLuts maxLutValue "<<maxLutValue<<std::endl;
    std::cout<<" maxFixedLutValue "<<maxFixedLutValue<<std::endl;


    for(unsigned int iInput = 0; iInput < lutArray.size(); iInput++) {
        unsigned int iNode = iInput;
        std::cout<<" iInput "<<std::setw(2)<<iInput<<" iNode "<<iNode<<std::endl;

        auto lutNode = static_cast<LutNode*>(layer.at(iNode).get() );

        if(lutArray.at(iInput).size() != lutNode->getFloatValues().size()) {
            std::cout<<"LutLayerFixedPoint::initLuts "<<name<<" lut size in lutArray"<<lutArray.at(iInput).size()
                                           <<" lutNode->getFloatValues().size() "<<lutNode->getFloatValues().size()<<" - not good !!!"<<std::endl;
            exit(1);
        }

        auto& lut = lutArray.at(iInput);
        for(unsigned int iAddr = 0; iAddr < lut.size(); iAddr++) {
            float value = lutNode->getFloatValues().at(iAddr);

            lut[iAddr] = ap_fixed<lut_W, lut_I,  AP_RND, AP_SAT>(value);

            if(value < -maxFixedLutValue ) {
                //std::cout<<" iNeuron "<<iNeuron<<" iInput "<<std::setw(2)<<iInput<<" iNode "<<std::setw(2)<< iNode<<" iAddr "<<std::setw(4)<<iAddr<<" value "<<std::setw(10)<<value<<" !!!!!!!!!!!!!!!!!!"<<std::endl;
            }
            else if(value > maxFixedLutValue) {
                //std::cout<<" iNeuron "<<iNeuron<<" iInput "<<std::setw(2)<<iInput<<" iNode "<<std::setw(2)<< iNode<<" iAddr "<<std::setw(4)<<iAddr<<" value "<<std::setw(10)<<value<<" maxFixedLutValue "<<maxFixedLutValue<<" !!!!!!!!!!!!!!!!!!"<<std::endl;
            }
        }
        //lut[lut.size() -1] = 0; ////FIXME for the moment ignoring the last addres, because in the layer 1 there is noHit value
    }

}

} /* namespace lutNN */

#endif /* INTERFACE_LUTLAYERFIXEDPOINT_H_ */
