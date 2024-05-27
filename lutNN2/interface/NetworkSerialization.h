/*
 * NetworkSerialization.h
 *
 *  Created on: Jan 15, 2020
 *      Author: kbunkow
 */

#ifndef INTERFACE_NETWORKSERIALIZATION_H_
#define INTERFACE_NETWORKSERIALIZATION_H_

#include "lutNN/lutNN2/interface/LutInterNetwork.h"
#include "lutNN/lutNN2/interface/NetworkBuilder.h"
#include "lutNN/lutNN2/interface/LutInter.h"
#include "lutNN/lutNN2/interface/LutBinary.h"
#include "lutNN/lutNN2/interface/SumNode.h"

namespace lutNN {

class NetworkSerialization {
public:
    NetworkSerialization();
    virtual ~NetworkSerialization();
};

/*template<class Archive>
void registerClasses(Archive & ar);*/

template<class Archive>
void registerClasses(Archive & ar) {
    ar.register_type(static_cast<LutNode*>(NULL));
    ar.register_type(static_cast<LutInter*>(NULL));
    ar.register_type(static_cast<InputNode*>(NULL));
    ar.register_type(static_cast<SumNode*>(NULL));
    ar.register_type(static_cast<SumIntNode*>(NULL));
    ar.register_type(static_cast<NetworkOutNode*>(NULL));
    ar.register_type(static_cast<SoftMax*>(NULL));
    ar.register_type(static_cast<SoftMaxWithSubClasses*>(NULL));
    ar.register_type(static_cast<LutBinary*>(NULL));
}

} /* namespace lutNN */

#endif /* INTERFACE_NETWORKSERIALIZATION_H_ */
