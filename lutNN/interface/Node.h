/*
 * Node.h
 *
 *  Created on: May 1, 2018
 *      Author: Karol Bunkowski
 */

#ifndef NODE_H_
#define NODE_H_

#include <vector>
#include <memory>

namespace lutNN {

class ConfigParameters {
public:
    enum LayerType {
        fullyConnected,
        singleLut,
        singleNetForOut,
        oneToOne //one node output is connected to only one node (i.e. LUT) of the next layer
    };

    enum LutType {
        discrete, //the LUT out value = val[addr]
        interpolated //the LUT out value is linear interpolation val[addr] and val[addr+1]
    };
    ConfigParameters() {};

    /**
     * @param lutAddrOffset - offset added to the neuron output value, should be half of the lutSize of the next layer (but in principle this can be a free parameter)
     */
    ConfigParameters(unsigned int neuronsInLayer, unsigned int lutSize, unsigned int nextLayerLutSize, unsigned int neuronInputCnt, LayerType layerType, LutType lutType ):
        neuronsInLayer(neuronsInLayer), lutSize(lutSize), nextLayerLutSize(nextLayerLutSize), neuronInputCnt(neuronInputCnt), layerType(layerType), lutType(lutType) {};

    unsigned int neuronsInLayer = 1;

    //number of LUT cell, i.e. maxAddres+1,
    unsigned int lutSize = 2<<6;

    unsigned int nextLayerLutSize = 2<<6;

    //unsigned int outValOffset = nextLayerLutSize/2;

    unsigned int neuronInputCnt = 2;

    LayerType layerType = fullyConnected;
    LutType lutType = discrete;
};

typedef std::shared_ptr<ConfigParameters> ConfigParametersPtr;

class Node {
public:
    enum NodeType {
        INPUT_NODE,
        NEURON_NODE
    };
    Node();
    virtual ~Node();

    //output value in form suitable for LUT addressing
    virtual int getOutAddr() {
        return outAddr;
    }

    float getOutVal() const {
        return outVal;
    }

    /*
     * changing the outAddr is needed for training
     */
    void setOutAddr(int outAddr) {
        this->outAddr = outAddr;
    }

    /*
     * changing the setOutVal is needed for training
     */
    void setOutVal(double outVal) {
        this->outVal = outVal;
    }

    virtual int getOutAddrUpdated() {
        return outAddrUpdated;
    }

    void setOutAddrUpdated(int outAddr) {
        this->outAddrUpdated = outAddr;
    }

    float getOutValUpdated() const {
        return outValUpdated;
    }

    bool wasOutAddrChanged() {
        return outAddrWasChanged;
    }

    virtual std::string name() const = 0;

    virtual NodeType getNodeType() const {
        return nodeType;
    }

    virtual void enable(bool enable) {
        enabled = enable;
    }

    virtual bool isEnabled() {
        return enabled;
    }

protected:
    int outAddr = 0;

    double outVal = 0;

    int outAddrUpdated = 0;

    bool outAddrWasChanged = true;

    double outValUpdated = 0;

    NodeType nodeType = INPUT_NODE;

    bool enabled = true;
};

class InputNode: public Node {
public:
    InputNode(unsigned int number);

    virtual ~InputNode();

    virtual void setInput(double value) {
        this->outAddr = value; //TODO add round if needed
        this->outVal = value;
    }

    virtual void setInput(std::vector<int> eventInputs) {
        this->outAddr = eventInputs[number]; //TODO add round if needed
        this->outVal = eventInputs[number];
    }

    virtual std::string name() const;
    friend std::ostream & operator << (std::ostream &out, const InputNode& node);

private:
    unsigned int number = 0;
};

class InputNodeSelInputs: public InputNode {
public:
    InputNodeSelInputs(unsigned int number, std::vector<int>& selectedInputs, std::vector<int>& selectedInputThresholds);

    virtual ~InputNodeSelInputs();

    virtual void setInput(std::vector<int> eventInputs);

    //friend std::ostream & operator << (std::ostream &out, const InputNode& node);

private:
    std::vector<int> selectedInputs;
    std::vector<int> selectedInputThresholds;
};

class InputNodeSelBinaryInputs: public InputNode {
public:
    InputNodeSelBinaryInputs(unsigned int number, std::vector<int>& selectedInputs);

    virtual ~InputNodeSelBinaryInputs() {};

    virtual void setInput(std::vector<int> eventInputs);

    //friend std::ostream & operator << (std::ostream &out, const InputNode& node);

private:
    std::vector<int> selectedInputs;
};

class NeuronNode; //forward declaration

class Lut {
public:
    //Lut() {};

    Lut(unsigned int lutSize, unsigned int number, NeuronNode* ownerNode);

    virtual ~Lut();

    //the pointer to the node from the previous layer
    virtual void setInputNode(InputNode* node);

    //the pointer to the node from the previous layer
    virtual void setInputNode(NeuronNode* node);

    //the pointer to the node from the previous layer
    virtual Node* getInputNode() {
        return this->inputNode;
    }

    //the pointer to the node that owns this LUT
    virtual NeuronNode* getOwnerNode() {
        return this->ownerNode;
    }

    //set the input addres for the curent event
    //virtual void setInputAddr(unsigned int addr);

    //sets the LUT output value for the input address taken from the input node
    virtual void run();

    //gives the difference between the current outVal (which should be set calling run() before,
    //and the value for the updated LUT input address from the input node
    virtual double update();

    //input address set in the last iteration, it is is assuered that it is in range 0...size-1
    int getAddr() const {
        return addr;
    }

    std::vector<float>& getValues() {
        return values;
    }

    virtual float getOutput() {
        return outVal;
    }

    struct LutStat {
        double gradientSumPos = 0; //deltaCost sum when the LUT value increased
        double gradientSumNeg = 0; //deltaCost sum when the LUT value decreased
        unsigned int entries = 0;

        double absGradientSum = 0;
        double momentum = 0;
    };

    std::vector<LutStat>& getLutStat() {
        return lutStat;
    }

    virtual std::string name() const;
    friend std::ostream & operator << (std::ostream &out, const Lut& l);


    virtual void setOutWeight(double outWeight) {
        this->outWeight = outWeight;
    }

    virtual double getDerivative() {
        return derivative;
    }
protected:
    std::vector<float> values;

    std::vector<LutStat> lutStat;

    //the pointer to the node from the previous layer
    Node* inputNode;

    int addr = 0;
    float outVal = 0;
    double derivative = 0;

    unsigned int number = 0;

    NeuronNode* ownerNode;

    //needed to implement dropout regularization
    double outWeight = 1;
};


class LutInter: public Lut {
public:
    //LutInter(): Lut() {};

    LutInter(unsigned int lutSize, unsigned int number, NeuronNode* ownerNode): Lut(lutSize, number, ownerNode) {};

    virtual ~LutInter() {};

    virtual void run();
};

typedef std::vector<std::unique_ptr<InputNode> > InputLayer;
class NeuronNode;
typedef std::unique_ptr<NeuronNode> NeuronNodePtr;
typedef std::vector<NeuronNodePtr > NeuronLayer;
typedef std::vector<NeuronLayer > NeuronLayersVec;

class LutNetwork;

class NeuronNode: public Node {
public:
    NeuronNode(std::shared_ptr<ConfigParameters> config, unsigned int layer, unsigned int number);

    virtual ~NeuronNode();

    void connectInput(InputNode* node, int iLut);
    void connectInputs(const InputLayer& nodes);

    void connectInput(const NeuronNodePtr& nodes, int iLut);
    void connectInputs(const NeuronLayer& nodes);

    virtual void run();

    virtual bool update(Lut* lut);

    virtual void resetUpdate();

    std::vector<std::unique_ptr<Lut> >& getLuts() {
        return luts;
    }

    std::vector<Lut*>& getChildLuts() {
        return childLuts;
    }

    void addChaildLut(Lut* lut) {
        childLuts.push_back(lut);
    }

    virtual std::string name() const;
    friend std::ostream & operator << (std::ostream &out, NeuronNode & node);

    unsigned int getNumber() const {
        return number;
    }

    unsigned int getLayer() const {
        return layer;
    }

    virtual void enable(bool enable) {
        enabled = enable;
        /*for(auto& lut : getChildLuts() ) {
            lut->setOutWeight(enable); //TODO remember that OutWeight is used to disable the childLut
        }*/
    }
/*    const float getDeltaCostMod(int addr) const {
        if(addr < 0)
            addr = 0;
        else if(addr >= deltaCostMod.size() )
            addr = deltaCostMod.size() -1;
        return deltaCostMod[addr];
    }*/

    friend LutNetwork;
private:
    std::shared_ptr<ConfigParameters> config;

    float outValOffset = 0;

    float outValOffsetStat = 0;

    int underflowOutCnt = 0;
    int overflowOutCnt = 0;

    int underflowOutValSum = 0;
    int overflowOutValSum = 0;

    int minOutAddr = config->nextLayerLutSize;
    int maxOutAddr = -1;

    std::vector<std::unique_ptr<Lut> > luts;

    //LUTs to which this node is connected
    //can be empty if this is the last layer
    std::vector<Lut*> childLuts;

    unsigned int layer = 0, number = 0;

    double derivative = 0;

};



} /* namespace lutNN */

#endif /* NODE_H_ */
