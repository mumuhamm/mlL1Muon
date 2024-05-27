/*
 * Node.cpp
 *
 *  Created on: May 1, 2018
 *      Author: Karol Bunkowski
 */

#include "lutNN/interface/Node.h"

#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>

namespace lutNN {

Node::Node() {
    // TODO Auto-generated constructor stub

}

Node::~Node() {
    // TODO Auto-generated destructor stub
}


InputNode::InputNode(unsigned int number): Node(), number(number) {
    nodeType = Node::INPUT_NODE;
}

InputNode::~InputNode() {
}

std::string InputNode::name()  const {
    return "InputNode " + std::to_string(number);
}

std::ostream & operator << (std::ostream &out, const InputNode& node) {
    out<<node.name()<<std::endl;

    return out;
}

////////////////////////////////////////////////////////////////////////


InputNodeSelInputs::InputNodeSelInputs(unsigned int number, std::vector<int>& selectedInputs, std::vector<int>& selectedInputThresholds):
        InputNode(number),
        selectedInputs(selectedInputs), selectedInputThresholds(selectedInputThresholds) {
}

InputNodeSelInputs::~InputNodeSelInputs() {
}

void InputNodeSelInputs::setInput(std::vector<int> eventInputs) {
    this->outAddr = 0;

    auto selectedInputThreshold = selectedInputThresholds.begin();
    unsigned int bit = 0;
    for(auto& selectedInput : selectedInputs) {
        if(eventInputs[selectedInput] >= *selectedInputThreshold) {
            this->outAddr |= (1 << bit);
        }
        bit++;
        selectedInputThreshold++;
    }

    this->outVal = this->outAddr;
}

InputNodeSelBinaryInputs::InputNodeSelBinaryInputs(unsigned int number, std::vector<int>& selectedInputs):
        InputNode(number),
        selectedInputs(selectedInputs) {
}


void InputNodeSelBinaryInputs::setInput(std::vector<int> eventInputs) {
    this->outAddr = 0;

    unsigned int bit = 0;
    for(auto& selectedInput : selectedInputs) {
        if(eventInputs[selectedInput]) {
            this->outAddr |= (1 << bit);
        }
        bit++;
    }

    this->outVal = this->outAddr;
}

////////////////////////////////////////////////////////////////////////

Lut::Lut(unsigned int lutSize, unsigned int number, NeuronNode* ownerNode): values(lutSize, 0), lutStat(lutSize), inputNode(0), number(number), ownerNode(ownerNode) {

}

Lut::~Lut() {
}

//the pointer to the node from the previous layer
void Lut::setInputNode(InputNode* node) {
    this->inputNode = node;
}

//the pointer to the node from the previous layer
void Lut::setInputNode(NeuronNode* node) {
    this->inputNode = node;
    node->addChaildLut(this );
}

/*void Lut::setInputAddr(unsigned int addr) {
	this->addr =  addr;
	outVal = lut.at(addr);
}*/

void Lut::run() {
    addr = inputNode->getOutAddr();
    if(addr < 0)
        addr = 0;
    else if(addr >= (int)values.size())
        addr = values.size() -1;
    outVal = values[addr] * outWeight;
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" addr "<<addr<<" outVal "<<outVal<<std::endl;
}

double Lut::update() {
    if(!outWeight)
        return 0;

    int addrUpdated = inputNode->getOutAddrUpdated();
    if(addrUpdated < 0)
        addrUpdated = 0;
    else if(addrUpdated >= (int)values.size())
        addrUpdated = values.size() -1;
    return (values[addrUpdated] - outVal);

    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" addr "<<addr<<" outVal "<<outVal<<std::endl;
}

void LutInter::run() {
    double inVal = inputNode->getOutVal();
    addr = floor(inVal );
    if(addr < 0) { //in principle can be merged with last else
        addr = 0;
        derivative = values[1] - values[0];
        outVal = values[0] + inVal * derivative;
    }
    else if(addr >= ((int)values.size() - 1) ) {
        addr = values.size() -1;
        derivative = values[addr] - values[addr -1];
        outVal = values[addr] + (inVal - addr) * derivative;
    }
    else {
        derivative = values[addr +1] - values[addr];
        outVal = values[addr] + (inVal - addr) * derivative;
    }

    if(isinf( outVal ) || isnan( outVal )) {
        std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<(*this)<<" outVal "<<outVal<<" derivative "<<derivative<<" addr "<<addr<<std::endl;
        exit(1);
    }
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" inVal "<<inVal<<" outVal "<<outVal<<std::endl;
}

std::string Lut::name()  const {
    return "LUT " + std::to_string(number);
}

std::ostream & operator << (std::ostream &out, const Lut & l) {
    out<<"LUT "<<std::setw(2)<<l.number<<" connected to "<<l.inputNode->name();
    //out<<std::endl;

    return out;
}

NeuronNode::NeuronNode(std::shared_ptr<ConfigParameters> config, unsigned int layer, unsigned int number):
        Node(), config(config), layer(layer), number(number)
    {
    nodeType = Node::NEURON_NODE;
    for(unsigned int iLut = 0; iLut < config->neuronInputCnt; iLut++) {
        if(config->lutType == ConfigParameters::discrete) //TDOO add config param
            luts.emplace_back(std::unique_ptr<Lut>(new Lut(config->lutSize, iLut, this)));
        else if(config->lutType == ConfigParameters::interpolated)
            luts.emplace_back(std::unique_ptr<Lut>(new LutInter(config->lutSize, iLut, this)));
        else
            throw std::invalid_argument("NeuronNode::NeuronNode: unknown LutType");
    }

    outValOffset = config->nextLayerLutSize / 2;
}

void NeuronNode::connectInput(InputNode* node, int iLut) {
    luts.at(iLut)->setInputNode(node);
}

void NeuronNode::connectInputs(const InputLayer& nodes) {
    if(nodes.size() != luts.size()) {
        std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<name()<<" nodes.size() != luts.size() "<<" nodes.size() "<<nodes.size()<<" luts.size() "<<luts.size()<<std::endl;
    }
    assert(nodes.size() == luts.size() ); //TODO change to throw

    for(unsigned int iLut = 0; iLut < luts.size(); iLut++) {
        luts[iLut]->setInputNode(nodes[iLut].get() );
    }
}

void NeuronNode::connectInput(const NeuronNodePtr& node, int iLut) {
    luts.at(iLut)->setInputNode(node.get() );
    //node->addChaildLut(luts.at(iLut).get() ); //todo maybe move to the Lut::setInputNode()???
}

void NeuronNode::connectInputs(const NeuronLayer& nodes) {
    assert(nodes.size() == luts.size() );
    for(unsigned int iLut = 0; iLut < luts.size(); iLut++) {
        luts[iLut]->setInputNode(nodes[iLut].get() );
        //nodes[iLut]->addChaildLut(luts[iLut].get() );
    }
}

NeuronNode::~NeuronNode() {

}

void NeuronNode::run() {
    outVal = outValOffset;
    /*if(!enabled) {
        outValUpdated = outVal;
        return;
    }*/

    for(auto& lut : luts) {
        lut->run();
        outVal += lut->getOutput();
    }
    outAddr = std::lround(outVal) ; //config->lutAddrOffset;
    outValUpdated = outVal;

    //debug
/*    if(outAddr < -100) {
        std::cout<<"NeuronNode::"<<__FUNCTION__<<":"<<__LINE__<<" neuron layer "<<layer <<" num "<<number<<" outVal "<<outVal<<" outAddr "<<outAddr<<std::endl;
        for(auto& lut : luts) {
            std::cout<<"    NeuronNode::"<<__FUNCTION__<<":"<<__LINE__<<" "<<lut->name()<<" addr "<<lut->getAddr()<<" outValue "<<lut->getOutput()<<std::endl;
        }
    }*/
}

bool NeuronNode::update(Lut* lut) {
    outValUpdated = outValUpdated + lut->update();

    //TODO can be done only once at the end
    outAddrUpdated = std::lround(outValUpdated) ; //config->lutAddrOffset;

    if(outAddrUpdated != outAddr)
        outAddrWasChanged = true;
    else
        outAddrWasChanged = false;

    return outAddrWasChanged;

    //debug
/*    if(outAddr < -100) {
        std::cout<<"NeuronNode::"<<__FUNCTION__<<":"<<__LINE__<<" neuron layer "<<layer <<" num "<<number<<" outVal "<<outVal<<" outAddr "<<outAddr<<std::endl;
        for(auto& lut : luts) {
            std::cout<<"    NeuronNode::"<<__FUNCTION__<<":"<<__LINE__<<" "<<lut->name()<<" addr "<<lut->getAddr()<<" outValue "<<lut->getOutput()<<std::endl;
        }
    }*/
}

void NeuronNode::resetUpdate() {
    outValUpdated = outVal;
    //outAddrUpdated = outAddr;
    //outAddrWasChanged = false; //TODO probably not needed
}

std::string NeuronNode::name()  const {
    return "Neuron: layer: " + std::to_string(layer) + " number " + std::to_string(number);
}

std::ostream & operator << (std::ostream &out, NeuronNode & node) {
    out<<node.name()<<std::endl;
    for(auto& lut : node.luts) {
        out<<"    "<<(*lut)<<std::endl;
    }

    return out;
}

} /* namespace lutNN */
