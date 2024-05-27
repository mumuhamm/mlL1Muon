/*
 * InputNodeFactory.h
 *
 *  Created on: Sep 18, 2018
 *      Author: kbunkow
 */

#ifndef INPUTNODEFACTORY_H_
#define INPUTNODEFACTORY_H_

#include <random>

#include "lutNN/lutNN2/interface/Node.h"

namespace lutNN {


class InputNodeFactory {
public:
    virtual ~InputNodeFactory() {};

    virtual InputNode* get(unsigned int number) = 0;
};

class InputNodeFactoryBase: public InputNodeFactory {

public:
    InputNodeFactoryBase() {};
    virtual ~InputNodeFactoryBase() {};

    virtual InputNode* get(unsigned int number) {
        return new InputNode(number);
    }
};

class InputNodeBinaryFactory: public InputNodeFactory {
public:
	/*
	 * if generator is nullptr -  the input node corresponding to the number is given
	 * Otherwise the random input is connected
	 */
    InputNodeBinaryFactory(unsigned int inputCnt, std::default_random_engine* generator = nullptr):
        inputCnt(inputCnt), generator(generator) {};

    virtual ~InputNodeBinaryFactory() {};

    virtual InputNode* get(unsigned int number);

private:
    unsigned int inputCnt = 0;

    std::default_random_engine* generator;
};

class InputNodeSelectedBitsFactory: public InputNodeFactory {
public:
	InputNodeSelectedBitsFactory(unsigned int inputCnt, std::default_random_engine& generator):
        inputCnt(inputCnt), generator(generator) {};

    virtual ~InputNodeSelectedBitsFactory() {};

    virtual InputNode* get(unsigned int number);

private:
    unsigned int inputCnt = 0;

    std::default_random_engine& generator;
};


class InputNodeSelInputsFactoryFlatDist: public InputNodeFactory {
public:
    InputNodeSelInputsFactoryFlatDist(unsigned int inputCnt, unsigned int selInputCnt, std::default_random_engine& generator);
    virtual ~InputNodeSelInputsFactoryFlatDist();

    virtual InputNode* get(unsigned int number);
private:
    unsigned int inputCnt = 0;
    unsigned int selInputCnt = 0;
    std::default_random_engine& generator;
};

class InputNodeSelBinaryInputsFactoryFlatDist: public InputNodeFactory {
public:
    InputNodeSelBinaryInputsFactoryFlatDist(unsigned int inputCnt, unsigned int selInputCnt, std::default_random_engine& generator);
    virtual ~InputNodeSelBinaryInputsFactoryFlatDist() {};

    virtual InputNode* get(unsigned int number);
private:
    unsigned int inputCnt = 0;
    unsigned int selInputCnt = 0;
    std::default_random_engine& generator;
};

/*
 class InputNodeSelInputsFactoryFlatDist {
public:
    InputNodeSelInputsFactoryFlatDist(unsigned int inputCnt, unsigned int selInputCnt, std::default_random_engine& generator, double radious);
    virtual ~InputNodeSelInputsFactoryFlatDist();

    InputNode* get(unsigned int number);
private:
    unsigned int selInputCnt = 0;
    std::default_random_engine& generator;
    double radious = 1;
};
 */

} /* namespace lutNN */

#endif /* INPUTNODEFACTORY_H_ */
