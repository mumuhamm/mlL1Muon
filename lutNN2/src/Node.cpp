/*
 * Node.cpp
 *
 *  Created on: Jul 8, 2019
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/Node.h"

#include <iostream>
#include <cassert>

namespace lutNN {

Node::Node(unsigned int number): number(number) {
    name = "Node_" + std::to_string(number);
}

Node::Node(unsigned int number, unsigned int inputCnt): number(number), inputNodes(inputCnt, nullptr) {
    name = "Node_" + std::to_string(number);
}

Node::~Node() {
    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
}

/*void Node::setInputNodes(InputLayer& nodes) {
    for(auto& node : nodes)
        inputNodes.push_back(node.get());
}*/

void Node::connectInput(NodePtr node, unsigned int index) {
    NodePtr previus = inputNodes.at(index);
    if(previus)
        std::cout<<"Warning: Node::connectInput node "<<getName() <<"- some not null point present at index "<<index<<std::endl;
    inputNodes.at(index) = node;
    //return previus;
}

void Node::connectInputs(NodeLayer& nodes) {
    assert(nodes.size() == inputNodes.size() );

    auto node = nodes.begin();
    for(auto& inputNode : inputNodes) {
        inputNode = node->get();
        node++;
    }
}

void SumIntNode::run(float eventWeight) {
    outValue = 0;
    for(auto& inNode : inputNodes) {
        outValue += round(inNode->getOutValue()); //TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< remove round
    }
}

InputNode::InputNode(): Node(0) {

}

InputNode::InputNode(unsigned int number): Node(number) {
    //nodeType = Node::INPUT_NODE;
}

InputNode::~InputNode() {
}

std::ostream & operator << (std::ostream &out, const InputNode& node) {
    out<<node.getName()<<std::endl;

    return out;
}

////////////////////////////////////////////////////////////////////////

void InputNodeSelectedBits::setOutValue(float value)  {
    this->outValue = (0b11000000 & ((int)value) ) >> 6;
}

//////////

InputNodeSelInputs::InputNodeSelInputs(unsigned int number, std::vector<int>& selectedInputs, std::vector<int>& selectedInputThresholds):
        InputNode(number),
        selectedInputs(selectedInputs), selectedInputThresholds(selectedInputThresholds) {
}

InputNodeSelInputs::~InputNodeSelInputs() {
}

void InputNodeSelInputs::setInput(std::vector<int>& eventInputs) {
    this->outValue = 0;

    auto selectedInputThreshold = selectedInputThresholds.begin();
    unsigned int bit = 0;
    for(auto& selectedInput : selectedInputs) {
        if(eventInputs[selectedInput] >= *selectedInputThreshold) {
            //this->outValue |= (1 << bit);
        	this->outValue = this->getOutValueInt() | (1 << bit);

        }
        bit++;
        selectedInputThreshold++;
    }

}

InputNodeSelBinaryInputs::InputNodeSelBinaryInputs(unsigned int number, std::vector<int>& selectedInputs):
        InputNode(number),
        selectedInputs(selectedInputs) {
}


void InputNodeSelBinaryInputs::setInput(std::vector<int>& eventInputs) {
    this->outValue = 0;

    unsigned int bit = 0;
    for(auto& selectedInput : selectedInputs) {
        if(eventInputs[selectedInput]) {
            //this->outValue |= (1 << bit);
            this->outValue = this->getOutValueInt() | (1 << bit);
        }
        bit++;
    }

}

} /* namespace lutNN */
