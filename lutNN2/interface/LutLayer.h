/*
 * LutLayer.h
 *
 *  Created on: Feb 6, 2022
 *      Author: kbunkow
 */

#ifndef INTERFACE_LUTLAYER_H_
#define INTERFACE_LUTLAYER_H_

#include <vector>
#include <limits>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/dynamic_bitset.hpp>


namespace lutNN {

template<typename OutValuesType>
class LutLayer {
public:
    LutLayer(size_t outputCnt, size_t lutCnt, size_t lutInputCnt):
        lutCnt(lutCnt), lutInputCnt(lutInputCnt), lastOutVals(outputCnt), floatLuts(lutCnt, 1<<lutInputCnt) {
    };

    virtual ~LutLayer() {};

    OutValuesType& getLastOutVals() {
        return lastOutVals;
    }

    virtual void calulateAddresses(boost::dynamic_bitset<>& inputs) {
        for(size_t iAddr = 0; iAddr < lastAddresses.size(); iAddr++) {
            lastAddresses[iAddr] = 0;
            for(size_t iPos = 0; iPos < lutInputCnt; iPos++) {
                if(inputs[ iAddr * lutInputCnt + iPos])
                    lastAddresses[iAddr] |= 1 << iPos;
            }
        }
    }

    void run(boost::dynamic_bitset<>& inputs);

    void updateGradients(std::vector<float>& prevLayerLastGradients);

protected:
    size_t lutCnt = 0;

    size_t lutInputCnt = 0;

    std::vector<unsigned short> lastAddresses;

    OutValuesType lastOutVals;

    //std::vector<float> floatLuts;

    boost::numeric::ublas::matrix<float> floatLuts;

    boost::numeric::ublas::matrix<float> gradients;

    boost::numeric::ublas::matrix<float> entries;

    std::vector<float> lastGradients;

};

typedef LutLayer<std::vector<float> > LutFloatLayer;


class LutBinaryLayer: public LutLayer<boost::dynamic_bitset<> > {
public:
    LutBinaryLayer(size_t outputCnt, size_t lutCnt, size_t lutInputCnt):
        LutLayer(outputCnt, lutCnt, lutInputCnt), binaryLuts(outputCnt, boost::dynamic_bitset<>(1<<lutInputCnt)) {
    }

    void run(boost::dynamic_bitset<>& inputs);

private:
    std::vector<boost::dynamic_bitset<> > binaryLuts;
};

} /* namespace lutNN */

#endif /* INTERFACE_LUTLAYER_H_ */
