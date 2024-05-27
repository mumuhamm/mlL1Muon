/*
 * ClassifierOutNode.h
 *
 *  Created on: Sep 15, 2019
 *      Author: kbunkow
 */

#ifndef NetworkOutNode_H_
#define NetworkOutNode_H_

#include "lutNN/lutNN2/interface/Node.h"
#include "lutNN/lutNN2/interface/CostFunction.h"

namespace lutNN {

class NetworkOutNode: public Node {
public:
    NetworkOutNode();

    NetworkOutNode(unsigned int classesCnt);

    virtual ~NetworkOutNode();

    virtual const std::vector<double>& getOutputValues() const {
        return outputValues;
    }

    //here just copies the values from the input nodes
    virtual void run(float eventWeight = 1);

    //each implementation of the NetworkOutNode should calculate the gradient for one given cost function
    //calcualteGradient should propagate the gradient back using Node::setGradientFromOutNode
    virtual void calcualteGradient(CostFunction& costFunction, std::vector<double> expextedResults, float eventWeight);

    virtual void calcualteGradient(CostFunction& costFunction, unsigned short& expectedClassLabel, float eventWeight);

    virtual const std::vector<double>& updateInput(unsigned int iClass, unsigned int iSubClass, double newInVal);

    virtual void print(std::ostream& ostr) const;
private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<Node>(*this);
        ar & outputValues;
        ar & modifiedOutputValues;
    }

protected:
    std::vector<double > outputValues;
    std::vector<double > modifiedOutputValues;
};


class SoftMax: public NetworkOutNode {
public:
    SoftMax();

    SoftMax(unsigned int classesCnt);

    virtual ~SoftMax() {};

    virtual void run(float eventWeight = 1);

    //version for the cross entropy error function!!!!!!
    virtual void calcualteGradient(CostFunction& costFunction, std::vector<double> expextedResults, float eventWeight);

    virtual void calcualteGradient(CostFunction& costFunction, unsigned short& expectedClassLabel, float eventWeight);

    static void softMax(std::vector<double>& inputValues, double& maxInVal, std::vector<double >& outputValues);

    void setMaxInVal(double maxInVal = 0) {
        this->maxInVal = maxInVal;
    }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<NetworkOutNode>(*this);
        ar & inputValues;
    }

private:
    //[iClass][iSubClass]
    //std::vector<NodeVec > inputNodes;

    std::vector<double> inputValues;

    double maxInVal = 0;
};


//for version with sub-classes
class SoftMaxWithSubClasses: public NetworkOutNode {
public:
    SoftMaxWithSubClasses();

    SoftMaxWithSubClasses(unsigned int classesCnt);

    virtual ~SoftMaxWithSubClasses() {};

    virtual void addSubClass(unsigned int classNum, NodePtr node) {
        inputNodesBySubClasses.at(classNum).emplace_back(node);
        inputValues.at(classNum).push_back(0);
    }

    virtual void run(float eventWeight = 1);

    //version for the cross entropy error function!!!!!!
    virtual void calcualteGradient(CostFunction& costFunction, std::vector<double> expextedResults, float eventWeight);

    virtual void calcualteGradient(CostFunction& costFunction, unsigned short& expectedClassLabel, float eventWeight);

    static void softMax(std::vector<std::vector<double> >& inputValues, double maxInVal, std::vector<double >& outputValues);


    virtual const std::vector<double>& updateInput(unsigned int iClass, unsigned int iSubClass, double newInVal);

    double getMaxInVal() const {
        return maxInVal;
    }

    void setMaxInVal(double maxInVal = 0) {
        this->maxInVal = maxInVal;
    }

    const std::vector<NodeVec>& getInputNodesBySubClasses() const {
        return inputNodesBySubClasses;
    }

    void print(std::ostream& ostr) const override;
/*
    const std::vector<std::vector<double> >& getInputValues() const {
        return inputValues;
    }
*/

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<NetworkOutNode>(*this);
        ar & inputNodesBySubClasses;
        ar & inputValues;
    }

private:
    //[iClass][iSubClass]
    std::vector<NodeVec > inputNodesBySubClasses;

    std::vector<std::vector<double> > inputValues;

    double maxInVal = 0;
};

} /* namespace lutNN */

#endif /* CLASSIFIEROUTNODE_H_ */
