/*
 * LutNode.h
 *
 *  Created on: Jan 16, 2022
 *      Author: kbunkow
 */

#ifndef INTERFACE_LUTNODE_H_
#define INTERFACE_LUTNODE_H_

#include "lutNN/lutNN2/interface/Node.h"

namespace lutNN {

class LutNode: public Node {
public:
    LutNode(): Node(0) {
        name = "LutNode_empty";
    }

    LutNode(unsigned int number, unsigned int lutSize, bool propagateGradient) :
        Node(number), propagateGradient(propagateGradient), floatValues(lutSize, 0), entries(lutSize, 0), gradients(lutSize, 0)
    {
        name = "LutNode_" + std::to_string(number);
    }

    LutNode(unsigned int number, unsigned int inputCnt, unsigned int lutSize, bool propagateGradient) :
        Node(number, inputCnt), propagateGradient(propagateGradient), floatValues(lutSize, 0), entries(lutSize, 0), gradients(lutSize, 0)
    {
        name = "LutNode_" + std::to_string(number);
    }

    //eventWeight is used to fill the entries
    virtual void run(float eventWeight = 1);

    void updateGradient() override;

    virtual void reset();

    virtual std::vector<float>& getFloatValues() {
        return floatValues;
    }

    //virtual std::vector<int>& getIntValues() = 0;

    virtual std::vector<float>& getEntries() {
        return entries;
    }

    virtual std::vector<float>& getGradients() {
        return gradients;
    }

    virtual boost::multi_array<float, 2>& getMomentum() {
        throw;
        //return momentum;
    }

    virtual unsigned int getLastAddr() const  {
        return lastAddr;
    }

    bool isDead() const {
        return dead;
    }

    void setDead(bool dead = false) {
        this->dead = dead;
    }

    virtual char getDither() const {
        return dither;
    }

    virtual void setDither(char dither) {
        this->dither = dither;
    }
    //virtual unsigned int getLastAddr() = 0;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<Node>(*this);

        ar & floatValues;

        if(entries.size() == 0) {
            entries.resize(floatValues.size(), 0.);
            gradients.resize(floatValues.size(), 0.);
        }
    }
protected:
    bool propagateGradient = true;

    //boost::multi_array<float, 2> momentum;
    std::vector<float> floatValues;

    std::vector<float> entries;

    std::vector<float> gradients;

    unsigned int lastAddr = 0;

    bool dead =  false;

    char dither = 0;
};
} /* namespace lutNN */



#endif /* INTERFACE_LUTNODE_H_ */
