/*
 * MNISTParser.cpp
 *
 *  Created on: Aug 20, 2018
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/MNISTParser.h"

// readMNIST.cc
// read MNIST data into double vector, OpenCV Mat, or Armadillo mat
// free to use this code for any purpose
// author : Eric Yuan
// my blog: http://eric-yuan.me/
// part of this code is stolen from http://compvisionlab.wordpress.com/
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int reverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void readMnistImages(string filename, vector<vector<uint8_t> >& images, unsigned int& rowCnt, unsigned int& columnCnt)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*) &rowCnt, sizeof(rowCnt));
        rowCnt = reverseInt(rowCnt);
        file.read((char*) &columnCnt, sizeof(columnCnt));
        columnCnt = reverseInt(columnCnt);
        for(int i = 0; i < number_of_images; ++i)
        {
            vector<uint8_t> tp;
            for(unsigned int r = 0; r < rowCnt; ++r)
            {
                for(unsigned int c = 0; c < columnCnt; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back(temp);
                }
            }
            images.push_back(tp);
        }
    }
}

void readMnistLabel(string filename, vector<uint8_t>& labels)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        labels.resize(number_of_images, 0);
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            labels[i]= temp;
        }
    }
}
