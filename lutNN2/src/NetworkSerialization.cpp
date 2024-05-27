/*
 * NetworkSerialization.cpp
 *
 *  Created on: Jan 15, 2020
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/NetworkSerialization.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

/*
#include "lutNN/lutNN2/interface/LutInterNetwork.h"
#include "lutNN/lutNN2/interface/NetworkBuilder.h"
#include "lutNN/lutNN2/interface/LutInter.h"
#include "lutNN/lutNN2/interface/LutNeuron.h"
*/

namespace lutNN {

NetworkSerialization::NetworkSerialization() {
    // TODO Auto-generated constructor stub

}

NetworkSerialization::~NetworkSerialization() {
    // TODO Auto-generated destructor stub
}

/* does not work for some reason
template<class Archive>
void registerClasses(Archive & ar) {
    ar.register_type(static_cast<LutInter*>(NULL));
    ar.register_type(static_cast<InputNode*>(NULL));
    ar.register_type(static_cast<LutNeuron*>(NULL));
    ar.register_type(static_cast<InputNode*>(NULL));
    ar.register_type(static_cast<SoftMax*>(NULL));
    ar.register_type(static_cast<SoftMaxWithSubClasses*>(NULL));
}

template<> void registerClasses(boost::archive::text_oarchive&);
template<> void registerClasses(boost::archive::text_iarchive&);*/

} /* namespace lutNN */
