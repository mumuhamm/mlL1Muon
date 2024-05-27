/*
 * Node.h
 *
 *  Created on: Jul 8, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_NODE_H_
#define INTERFACE_NODE_H_

#include <vector>
#include <memory>

#include <iostream>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/assume_abstract.hpp>
//#include <boost/serialization/export.hpp>

#include "boost/multi_array.hpp"


namespace lutNN {

class Node;

typedef Node* NodePtr;

typedef std::vector<NodePtr> NodeVec;

typedef std::vector<std::unique_ptr<Node> > NodeLayer;
//typedef std::vector<std::shared_ptr<Node> > NodeLayer;


class InputNode;

typedef std::vector<std::unique_ptr<Node> > InputLayer;

struct LearnigParams {
    double learnigRate = 0;
    double beta = 0; //momentum rate
    double lambda = 0; //l2 regularization rate
    double stretchRate = 0;
    //double regularizationRate = 0;
    double smoothWeight = 0;
    //float maxLutVal = 1;
};

//todo make the node template of the output value, to allow for int or float or double values
class Node {
public:
    Node(unsigned int number);

    Node(unsigned int number, unsigned int inputCnt);

    virtual ~Node();

    const NodeVec& getInputNodes() const {
        return inputNodes;
    }

    /*
    void setInputNodes(const std::vector<NodePtr>& inputNodes) {
        this->inputNodes = inputNodes;
    }

    void setInputNodes(NodeLayer& nodes) {
        for(auto& node : nodes)
            inputNodes.push_back(node.get());
    }

    void setInputNodes(InputLayer& nodes);
     */

    virtual void connectInput(NodePtr node, unsigned int index);

    void connectInputs(NodeLayer& nodes);

    void addInput(NodePtr inNode) {
        inputNodes.push_back(inNode);
    }

    /*    const std::vector<NodePtr>& getOutputNodes() const {
        return outputNodes;
    }

    void setOutputNodes(const std::vector<NodePtr>& outputNodes) {
        this->outputNodes = outputNodes;
    }*/

    virtual void setOutValue(float value) {
        this->outValue = value;
    }

    virtual void run(float eventWeight = 1) = 0;

    virtual void updateStat(unsigned int eventWeihgt, int outValue, float cost) {}

    virtual float& getOutValue() {
        return outValue;
    }

    virtual int getOutValueInt() {
        return this->outValue; //FIXME shouldn't be round?
    }


    virtual float& getLastGradient() {
        return lastGradient;
    }

    //Default implementation - does nothing but propagating back the lastGradient
    virtual void updateGradient() {
        for(auto& inputNode : inputNodes) {
            inputNode->getLastGradient() += lastGradient;
        }
        lastGradient = 0;
    }

    //for training
    virtual void updateParamaters(LearnigParams& learnigParams) {}

    virtual std::string getName() const {
        return name;
    }

    virtual void setName(std::string name) {
        this->name =  name;
    }

    virtual void reset() {};

    virtual char getDither() const {
        return 0;
    }

    virtual void setDither(char dither) {
    }

    unsigned int getNumber() const {
        return number;
    }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & number;
        ar & inputNodes;
        ar & name;
    }

protected:
    unsigned int number = 0;

    NodeVec inputNodes;
    //std::vector<NodePtr> outputNodes;

    float outValue = 0;

    float lastGradient = 0;

    std::string name;
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT( Node );


class SumIntNode: public Node {
public:
    SumIntNode(): Node(0, 0) {
        name = "SumNode_empty";
    }

    SumIntNode(unsigned int number, unsigned int inputCnt): Node(number, inputCnt) {
    }


    virtual void run(float eventWeight = 1);

    /*    virtual int getOutValueInt() {
        return this->outValue;
    }*/


private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<Node>(*this);
    }

private:

};




class InputNode: public Node {
public:
    InputNode();

    InputNode(unsigned int number);

    virtual ~InputNode();

    virtual void run(float eventWeight = 1) {}

    virtual std::string getName()  const {
        return "InputNode_" + std::to_string(number);
    }

    friend std::ostream & operator << (std::ostream &out, const InputNode& node);

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<Node>(*this);
    }

protected:
    //float outValue = 0;
};

//BOOST_CLASS_EXPORT_GUID(InputNode, "InputNode");

class InputNodeBinary: public InputNode {
public:
    InputNodeBinary(unsigned int number, unsigned int inputNum, float threshold): InputNode(number), inputNum(inputNum), threshold(threshold) {

    }

    virtual ~InputNodeBinary() {};

    void setOutValue(float value) override {
        if(value >= threshold)
            this->outValue = 1;
        else
            this->outValue = 0;

        deltaValThresh = value - threshold;
    }

    //virtual void run() {}

    virtual std::string getName()  const {
        return "InputNodeBinary_" + std::to_string(number) + "_inputNum_" + std::to_string(inputNum);
    }

    virtual float& getThreshold() {
        return threshold;
    }

    virtual float& getGradient() {
        return gradient;
    }

    virtual float getDeltaValThresh() {
        return deltaValThresh;
    }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<InputNode>(*this);
        ar & inputNum;
        ar & threshold;
    }

protected:
    //int outValue = 0;
    unsigned int inputNum = 0;
    float threshold = 64;
    float gradient = 0;

    float deltaValThresh = 0;
};

class InputNodeSelectedBits: public InputNode {
public:
    InputNodeSelectedBits(unsigned int number, unsigned int inputNum): InputNode(number), inputNum(inputNum) {

    }

    virtual ~InputNodeSelectedBits() {};


    void setOutValue(float value) override ;

    //virtual void run() {}

    virtual std::string getName()  const {
        return "InputNodeBinary_" + std::to_string(number) + "_inputNum_" + std::to_string(inputNum);
    }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<InputNode>(*this);
        ar & inputNum;
    }

protected:
    //int outValue = 0;
    unsigned int inputNum = 0;
};

class InputNodeSelInputs: public InputNode {
public:
    InputNodeSelInputs(unsigned int number, std::vector<int>& selectedInputs, std::vector<int>& selectedInputThresholds);

    virtual ~InputNodeSelInputs();

    virtual void setInput(std::vector<int>& eventInputs);

    virtual std::string getName()  const {
        return "InputNodeSelInputs_" + std::to_string(number);
    }

    /*
    virtual int getOutValueInt() {
        return this->outValue;
    }
     */

    //friend std::ostream & operator << (std::ostream &out, const InputNode& node);

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<InputNode>(*this);
        ar & selectedInputs;
        ar & selectedInputThresholds;
    }

private:
    std::vector<int> selectedInputs;
    std::vector<int> selectedInputThresholds;

    //int outValue = 0;
};

class InputNodeSelBinaryInputs: public InputNode {
public:
    InputNodeSelBinaryInputs(unsigned int number, std::vector<int>& selectedInputs);

    virtual ~InputNodeSelBinaryInputs() {};

    virtual void setInput(std::vector<int>& eventInputs);

    //friend std::ostream & operator << (std::ostream &out, const InputNode& node);

    /*    virtual int getOutValueInt() {
        return this->outValue;
    }*/

    virtual std::string getName()  const {
        return "InputNodeSelBinaryInputs_" + std::to_string(number);
    }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<InputNode>(*this);
        ar & selectedInputs;
    }

private:
    std::vector<int> selectedInputs;
    //int outValue = 0;
};

} /* namespace lutNN */

#endif /* INTERFACE_NODE_H_ */
