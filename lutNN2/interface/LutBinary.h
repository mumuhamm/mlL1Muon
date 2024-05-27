/*
 * LutNode.h
 *
 *  Created on: Jul 8, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_LUTBINARY_H_
#define INTERFACE_LUTBINARY_H_

#include <vector>
#include <memory>
#include <boost/dynamic_bitset.hpp>

#include "lutNN/lutNN2/interface/LutNode.h"

//#include "boost/multi_array.hpp"

namespace boost { namespace serialization {

    template <typename Ar, typename Block, typename Alloc>
        void save(Ar& ar, dynamic_bitset<Block, Alloc> const& bs, unsigned) {
            size_t num_bits = bs.size();
            std::vector<Block> blocks(bs.num_blocks());
            to_block_range(bs, blocks.begin());

            ar & num_bits & blocks;
        }

    template <typename Ar, typename Block, typename Alloc>
        void load(Ar& ar, dynamic_bitset<Block, Alloc>& bs, unsigned) {
            size_t num_bits;
            std::vector<Block> blocks;
            ar & num_bits & blocks;

            bs.resize(num_bits);
            from_block_range(blocks.begin(), blocks.end(), bs);
            bs.resize(num_bits);
        }

    template <typename Ar, typename Block, typename Alloc>
        void serialize(Ar& ar, dynamic_bitset<Block, Alloc>& bs, unsigned version) {
            split_free(ar, bs, version);
        }

} }

namespace lutNN {

//todo move to the LutBinary, entries are already in LutNode
struct LutStat {
    LutStat(unsigned int outputBits)//: costVsOutVal(1<<outValCnt, 0)
    {
        costVsOutVal = new float[1<<outputBits];
        for(unsigned int i = 0; i < (1u<<outputBits); i++)
            costVsOutVal[i] = 0;
    }

    unsigned int entries = 0;
    //std::vector<float> costVsOutVal; //index is the output value - makes sense only if the output value has not more than a few bits
    float* costVsOutVal; //vector is to costly for such a small arrays
};


/**
 * LUT module with multiple inputs - each with bitsPerInput,
 * and integer values - in principle for all layers but last only binary value are possible otherwise the training does not work
 * floatValues are needed only for training
 */
class LutBinary: public LutNode {
public:
    LutBinary();

    LutBinary(unsigned int number, unsigned int bitsPerInput, unsigned int inputCnt, unsigned int outputBits, bool propagateGradient);
    virtual ~LutBinary();

    virtual void run(float eventWeight = 1);

    //std::vector<bool>& getIntValues()
    boost::dynamic_bitset<>& getIntValues()  {
        return values;
    }

    void setValue(unsigned int address, bool value) {
        values[address] = value;
    }

    virtual void updateGradient();

    virtual void updateStat(unsigned int eventWeihgt, int outVal, float cost);

    //virtual void resetStat();


    //typedef boost::multi_array<float, 2> gradientsArrayType;
    //typedef std::vector<float> gradientsArrayType;

    /*gradientsArrayType& getGradients() {
        return gradients;
    }*/

    /*std::vector<float>& getLastGradientVsOutVal() {
        return lastGradient;
    }*/


    /*virtual unsigned int getLastAddr() const {
        return lastAddr;
    }*/

    /*virtual std::vector<float>& getFloatValues() {
        return floatValues;
    }*/

    /*virtual std::vector<float>& getEntries() {
        return entries;
    }
*/
protected:
    //std::vector<bool> values;
    boost::dynamic_bitset<> values;

    //std::vector<LutNodePtr> inputNodes;
    //std::vector<LutNodePtr> outputNodes;

    //int outValue = 0; //outValue from last run()

    unsigned int bitsPerInput = 1; //number of bits from each input used to build input address

    unsigned int outputBits = 1;

    //for trainig
    //std::vector<float> floatValues; //LUT output values in float for
    //gradientsArrayType gradients; //[address] gradients versus LUT address
    //std::vector<float> lastGradient; //[outValue]

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<LutNode>(*this);

        ar & values;

        ar & bitsPerInput;
        ar & outputBits;
    }

};


} /* namespace lutNN */

#endif /* INTERFACE_LUTBINARY_H_ */
