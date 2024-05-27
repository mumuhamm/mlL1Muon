/*
 * CostFunction.cpp
 *
 *  Created on: Feb 7, 2021
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/CostFunction.h"

namespace lutNN {

double CostFunction::get(std::vector<double> expectedResults, std::vector<double> nnResults)  {
    unsigned short expectedClassLabel = 0;
    for(unsigned int iClass = 0; iClass < expectedResults.size(); iClass++) {
        if(expectedResults[iClass] != 0) {
            expectedClassLabel = iClass;
            //break; cannot break, because the derivative will not be cleaned
        }
    }
    return get(expectedClassLabel, nnResults) ;
}

double CostFunction::get(unsigned short& expectedClassLabel, std::vector<double> nnResults)  {
    derivative.resize(nnResults.size());
    double cost = 0;

    auto& resultForExpectedClass = nnResults[expectedClassLabel];
    for(unsigned int iClass = 0; iClass < nnResults.size(); iClass++) {
        if(iClass == expectedClassLabel) {
            derivative[iClass] = -1;
        }
        else {
            derivative[iClass] = 1;

            if(resultForExpectedClass < nnResults[iClass])
                cost += 1.;
        }
    }

    return cost;
}

double CostFunctionMeanSquaredError::get(std::vector<double> expectedResults, std::vector<double> nnResults)  {
    derivative.resize(nnResults.size());
    double cost = 0;
    auto expectedResult = expectedResults.begin();
    unsigned int iClass = 0;
    for(auto& nnResult : nnResults) {
        cost += pow(*expectedResult - nnResult, 2);
        expectedResult++;

        derivative[iClass] = 2. * (nnResults[iClass] - expectedResults[iClass]);
        iClass++;
    }
    return cost;
}

double CostFunctionMeanSquaredError::get(unsigned short& expectedClassLabel, std::vector<double> nnResults)  {
    derivative.resize(nnResults.size());
    double cost = 0;

    for(unsigned int iClass = 0; iClass < nnResults.size(); iClass++) {
        double expectedResult = 0;
        if(iClass == expectedClassLabel) {
            expectedResult =  1;
        }
        cost += pow(expectedResult - nnResults[iClass], 2);

        derivative[iClass] = 2. * (nnResults[iClass] - expectedResult);
    }

    return cost;
}

double CostFunctionAbsoluteError::get(std::vector<double> expectedResults, std::vector<double> nnResults)  {
    derivative.resize(nnResults.size());
    double cost = 0;
    /*auto expectedResult = expectedResults.begin();
    unsigned int iClass = 0;
    for(auto& nnResult : nnResults) {
        double diff = nnResult - *expectedResult;
        cost += fabs(diff);
        expectedResult++;

        if(diff > 0)
            derivative[iClass] = 1;
        else if(diff < 0)
            derivative[iClass] = -1;
        iClass++;
    }*/

    for(unsigned int iOut = 0; iOut < expectedResults.size(); iOut++) {
        double diff = nnResults.at(iOut) - expectedResults[iOut];
        cost += fabs(diff);
        if(diff > 0) {
            derivative.at(iOut) = 1;
        }
        else {
            derivative.at(iOut) = -1;
        }
    }

    return cost;
}

double CostFunctionAbsoluteError::get(unsigned short& expectedClassLabel, std::vector<double> nnResults)  {
    derivative.resize(nnResults.size());
    double cost = 0;

    for(unsigned int iOut = 0; iOut < nnResults.size(); iOut++) {
        double expectedResult = 0;
        if(iOut == expectedClassLabel)
            expectedResult = 1;

        double diff = nnResults.at(iOut) - expectedResult;

        cost += fabs(diff);
        if(diff > 0) {
            derivative.at(iOut) = 1;
        }
        else {
            derivative.at(iOut) = -1;
        }
    }

    return cost;
}


double CostFunctionCrossEntropy::get(unsigned short& expectedClassLabel, std::vector<double> nnResults)  {
    return -log(nnResults[expectedClassLabel]);
}

double CostFunctionHingeLost::get(unsigned short& expectedClassLabel, std::vector<double> nnResults)  {
    derivative.resize(nnResults.size());

    auto& resultForExpectedClass = nnResults[expectedClassLabel];

    double cost = 0;
    derivative[expectedClassLabel] = 0;

    for(unsigned int iClass = 0; iClass < nnResults.size(); iClass++) {
        if(iClass != expectedClassLabel) {
            double diff = nnResults[iClass] - resultForExpectedClass + margin;
            if(diff > 0) {
                //cost += diff;
                cost += 1.;
                derivative[iClass] = 1;
                derivative[expectedClassLabel] -= 1;
            }
            else {
                //derivative[iClass] = 0;
                derivative[iClass] = 1. / (resultForExpectedClass - nnResults[iClass]);
                derivative[expectedClassLabel] -=  1. / (resultForExpectedClass - nnResults[iClass]);
            }
        }
    }
    return cost;
}

} /* namespace lutNN */
