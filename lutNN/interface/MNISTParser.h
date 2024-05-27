/*
 * MNISTParser.h
 *
 *  Created on: Aug 20, 2018
 *      Author: kbunkow
 */

#ifndef MNISTPARSER_H_
#define MNISTPARSER_H_

#include <string>
#include <vector>

void readMnistImages(std::string filename, std::vector<std::vector<uint8_t> >& images, unsigned int& rowCnt, unsigned int& columnCnt);

void readMnistLabel(std::string filename, std::vector<uint8_t>& labels);

#endif /* MNISTPARSER_H_ */
