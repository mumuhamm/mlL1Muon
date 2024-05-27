/*
 * SumNode.h
 *
 *  Created on: Dec 13, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_SUMNODE_H_
#define INTERFACE_SUMNODE_H_

#include "lutNN/lutNN2/interface/Node.h"
#include "lutNN/lutNN2/interface/LutInter.h"
#include "lutNN/lutNN2/interface/LayerConfig.h"

namespace lutNN {

/*
 * SumNode is a simple node that just does the sum of its input
 */
class SumNode: public Node {
public:
    SumNode();

    SumNode(unsigned int number, unsigned int inputCnt, float outValOffset, unsigned int biasShift, bool shiftLastGradient);
    virtual ~SumNode();

    virtual void run(float eventWeight = 1);

/*    std::vector<LutNode*>& getChildLuts() {
        return childLuts;
    }

    void addChildLut(LutNode* lut) {
        childLuts.push_back(lut);
    }*/

    virtual void updateGradient();

    void updateParamaters(LearnigParams& learnigParams) override;

    virtual void reset();

    float getOutValOffset() const {
        return outValOffset;
    }

    void setOutValOffset(float outValOffset = 0) {
        this->outValOffset = outValOffset;
    }

    void setBiasNode(NodePtr biasNode) {
        this->biasNode = biasNode;
    }

    void setBatchNormOffset(double batchNormOffset = 0) {
        this->batchNormOffset = batchNormOffset;
    }

    void setBatchNormScale(double batchNormScale = 0) {
        this->batchNormScale = batchNormScale;
    }

    double getOutVal2Sum() const {
        return outVal2Sum;
    }

    double getOutValSum() const {
        return outValSum;
    }

    virtual std::string print();
private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<Node>(*this);
        ar & biasNode;
        ar & outValOffset;
        ar & biasShift;
        ar & batchNormScale;
        ar & batchNormOffset;
    }

private:
    //LayerConfig* layerConfig = nullptr;
    NodePtr biasNode = nullptr;

    float outValOffset = 0;
    unsigned int biasShift = 0;

    bool shiftLastGradient = false;

    //LUTs to which this node is connected
    //can be empty if this is the last layer
    //std::vector<LutNode*> childLuts; //TODO maybe rather should be nodes?

    //for batch normalization
    //
    float inValSum = 0;

    float shiftFactor = 0;

    double outValSum = 0;
    double outVal2Sum = 0;

    float maxInValSum = 0;
    float minInValSum = 0;

    double batchNormScale = 1;
    double batchNormOffset = 0;

};

} /* namespace lutNN */

#endif /* INTERFACE_SUMNODE_H_ */
