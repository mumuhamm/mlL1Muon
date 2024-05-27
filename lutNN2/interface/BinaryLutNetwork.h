/*
 * BinaryLutNetwork.h
 *
 *  Created on: Jul 10, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_BINARYLUTNETWORK_H_
#define INTERFACE_BINARYLUTNETWORK_H_

#include "lutNN/lutNN2/interface/LutBinary.h"
#include "lutNN/lutNN2/interface/LutNetworkBase.h"
#include "lutNN/lutNN2/interface/NetworkOutNode.h"
#include "lutNN/lutNN2/interface/CostFunction.h"

namespace lutNN {

class BinaryLutNetwork: public LutNetworkBase {
public:
    BinaryLutNetwork();

    BinaryLutNetwork(NetworkOutNode* outputNode, std::default_random_engine* rndGenerator);

    virtual ~BinaryLutNetwork();

    virtual void initLutsRnd(std::default_random_engine& generator);

    virtual void initLutsAnd(std::default_random_engine& generator);

    void run(float eventWeight = 1) override;

    virtual void runTraining(EventInt* event, CostFunction& costFunction);

    //virtual void runTrainingAndUpdate(EventInt* event, CostFunction& costFunction, std::vector<LearnigParams>& learnigParamsVec);

    virtual void updateLuts(std::vector<LearnigParams>& learnigParamsVec);

    //virtual void dither(std::vector<LearnigParams>& learnigParamsVec, std::default_random_engine& rndGenerator);

    //has sense only if a given LutNode is only in one branch
    virtual void runTrainingNaiveBayes(std::vector<double> expextedResult);

    virtual void updateLutsNaiveBayes(unsigned int iLayer);


    typedef std::vector<std::vector<std::vector<std::vector<NodePtr> > > > Branches;
    Branches& getBranches() {
        return branches;
    }

    /*virtual const std::vector<double>& getOutputValues() {
        return outputNode->getOutputValues();
    }*/
protected:
    Branches branches; //[outNum][branchNum][layer]

    std::default_random_engine* rndGenerator = nullptr;

    //virtual void calcualteOutputValues();


private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<LutNetworkBase>(*this);

        ar & (branches);
    }

};

} /* namespace lutNN */

#endif /* INTERFACE_BINARYLUTNETWORK_H_ */
