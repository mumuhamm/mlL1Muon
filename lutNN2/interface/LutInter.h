/*
 * LutInter.h
 *
 *  Created on: Dec 13, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_LUTINTER_H_
#define INTERFACE_LUTINTER_H_

#include <vector>
#include <memory>

#include "lutNN/lutNN2/interface/LutNode.h"

namespace lutNN {

/**
 * LUT module with output value that is interpolated from the values of the two neighboring addresses
 * only one input is possible
 *
 * TODO move the below to LutInter2In
 * multiple inputs are possible, then the address is composed from the bits of the inputs,
 * but all the inputs but [0] must be InputNodes, otherwise the training will not work
 */
class LutInter: public LutNode {
public:
    LutInter(); //needed for serialization

    LutInter(unsigned int number, unsigned int bitsPerInput, bool propagateGradient, unsigned int rangesCnt = 1, bool interpolate = true);

    virtual ~LutInter();

    virtual void run(float eventWeight = 1);

    /*virtual std::vector<float>& getFloatValues() {
        return floatValues;
    }*/

    virtual std::vector<int>& getIntValues() {
        throw std::runtime_error("LutInter::getIntValues() not implemented!!");
    }

    virtual unsigned int getLastAddr() const {
        return lastAddr;
    }

   /* virtual float getOutValue() {
        return outValue;
    }*/

    void updateGradient() override;

    virtual void reset();

    virtual void connectInput(NodePtr node, unsigned int index) {
        //todo add throw if index != 0
        addInput(node);
        inputNode = node;
        //return Node::connectInput(node,  index);
    }

    virtual std::vector<float>& getMomementum() {
        return momementum;
    }

    virtual std::vector<float>& getEntries() {
        return entries;
    }

    unsigned int getRangesCnt() const {
        return rangesCnt;
    }

    void setRangesCnt(unsigned int rangesCnt = 1) {
        this->rangesCnt = rangesCnt;
    }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<LutNode>(*this);

        ar & rangesCnt;

        ar & interpolate;

        //ar & floatValues;
        //ar & entries; //the actual values are not needed to be stored, but the vector must be initialized somehow to the proper length
        //ar & gradients;
        //ar & momementum;

        ar & inputNode;
        ar & outputNode;

        if(momementum.size() == 0) {
            momementum.resize(floatValues.size(), 0.);
        }
    }

private:
    /**
     the LUT can be divided into ranges of equal size, there should be not interpolation between the ranges edges,
     i.e. e.g. there should be no input value of 127.2 in case of rangesCnt = 8 and lutSize = 1024
     */
    unsigned int rangesCnt = 1;

    //if fasle, no interpolation is used in run
    bool interpolate = true;

    std::vector<float> momementum;

    Node* inputNode = nullptr; //maybe rsther remove and use only Node::inputNodes

    Node* outputNode = nullptr; //in principle can be only sum node


    float derivative = 0;
};

} /* namespace lutNN */

#endif /* INTERFACE_LUTINTER_H_ */
