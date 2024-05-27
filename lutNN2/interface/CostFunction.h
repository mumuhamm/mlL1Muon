/*
 * CostFunction.h
 *
 *  Created on: Feb 7, 2021
 *      Author: kbunkow
 */

#ifndef INTERFACE_COSTFUNCTION_H_
#define INTERFACE_COSTFUNCTION_H_

#include "lutNN/lutNN2/interface/Event.h"
#include "lutNN/lutNN2/interface/Node.h"

namespace lutNN {

class CostFunction {
public:
    //TODO add constructor and initailise derivative
    virtual ~CostFunction() {};

    virtual double get(std::vector<double> expextedResults, std::vector<double> nnResults);

    virtual double get(unsigned short& expectedClassLabel, std::vector<double> nnResults);

    virtual double operator() (std::vector<double> expextedResults, std::vector<double> nnResults) {
        return get(expextedResults, nnResults);
    }

    virtual std::vector<double>& getDerivative() {
        return derivative;
    }
    //virtual double derivative(std::vector<double> expextedResults, std::vector<double> nnResults, unsigned int outNum)  = 0;
    //gradient is calculated by the NetworkOutNode

protected:
    std::vector<double> derivative;
};

class CostFunctionMeanSquaredError: public  CostFunction {
public:
    virtual ~CostFunctionMeanSquaredError() {};

    virtual double get(std::vector<double> expextedResults, std::vector<double> nnResults);

    virtual double get(unsigned short& expectedClassLabel, std::vector<double> nnResults);

/*    virtual double derivative(std::vector<double> expextedResults, std::vector<double> nnResults, unsigned int outNum) {
        //return -2. * (expextedResults[outNum ] - nnResults[outNum] );
        return 2. * (nnResults[outNum] - expextedResults[outNum ]);
    }*/

};

class CostFunctionAbsoluteError: public  CostFunction {
public:
    virtual ~CostFunctionAbsoluteError() {};

    virtual double get(std::vector<double> expextedResults, std::vector<double> nnResults);

    virtual double get(unsigned short& expectedClassLabel, std::vector<double> nnResults);

/*    virtual double derivative(std::vector<double> expextedResults, std::vector<double> nnResults, unsigned int outNum) {
        //return -2. * (expextedResults[outNum ] - nnResults[outNum] );
        return 2. * (nnResults[outNum] - expextedResults[outNum ]);
    }*/

};

class CostFunctionCrossEntropy: public  CostFunction {
public:
    virtual ~CostFunctionCrossEntropy() {};

    //virtual double get(std::vector<double> expextedResults, std::vector<double> nnResults);

    virtual double get(unsigned short& expectedClassLabel, std::vector<double> nnResults);

    //if the output layer is softmax!!!!
/*
    virtual double derivative(std::vector<double> expextedResults, std::vector<double> nnResults, unsigned int outNum) {
        return nnResults[outNum] - expextedResults[outNum ];
    }
*/

};

class CostFunctionHingeLost: public  CostFunction {
public:
    double margin = 1;

    virtual ~CostFunctionHingeLost() {};

    //virtual double get(std::vector<double> expextedResults, std::vector<double> nnResults);

    virtual double get(unsigned short& expectedClassLabel, std::vector<double> nnResults);

private:

};


} /* namespace lutNN */

#endif /* INTERFACE_COSTFUNCTION_H_ */
