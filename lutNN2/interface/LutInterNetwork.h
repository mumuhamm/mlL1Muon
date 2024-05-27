/*
 * LutInterNetwork.h
 *
 *  Created on: Dec 13, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_LUTINTERNETWORK_H_
#define INTERFACE_LUTINTERNETWORK_H_

#include "lutNN/lutNN2/interface/LutNetworkBase.h"
#include "NetworkOutNode.h"

namespace lutNN {

class LutInterNetwork: public LutNetworkBase {
public:
	LutInterNetwork(): LutNetworkBase() {};

    LutInterNetwork(NetworkOutNode* outputNode, bool useNoHitCntNode);
    virtual ~LutInterNetwork();

    virtual void updateLuts(std::vector<LearnigParams>& learnigParamsVec);

    void smoothLuts(std::vector<LearnigParams>& learnigParamsVec);

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<LutNetworkBase>(*this);
    }
};

} /* namespace lutNN */

#endif /* INTERFACE_LUTINTERNETWORK_H_ */
