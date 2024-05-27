/*
 * LayerConfig.h
 *
 *  Created on: Mar 31, 2021
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef INTERFACE_LAYERCONFIG_H_
#define INTERFACE_LAYERCONFIG_H_

#include <vector>
#include <memory>
#include <string>

namespace lutNN {

class LayerConfig {
public:

    virtual ~LayerConfig();
    unsigned int nodesInLayer = 1;

    enum NodeType {
        softMax,
        sumIntNode, //TODO merge with sumNode
        sumNode,
        lutNode,
        lutInter,
        lutBinary
        //only nodes that are derived from LutNode should be after lutNode, because LutNetworkPrint::printLuts2 uses >
    };
    NodeType nodeType;

    static std::string nodeTypeToStr(NodeType nodeType);

    unsigned int nodeInputCnt = 1;

    unsigned int classesCnt = 1;
    unsigned int subClassesCnt = 1;

    unsigned int bitsPerNodeInput = 1;

    unsigned int outputBits = 1;

    unsigned int lutRangesCnt = 1;
    unsigned int lutBinsPerRange = 0;

    bool interpolate = true;

    //if true, the last LUT address is reserver for the noHit
    bool noHitValue = false;

    bool propagateGradient = true;

    float maxLutVal = 0; //max value
    float minLutVal = 0; //max value

    float middleLutVal = 0; //middle absolute value

    float maxLutValChange = 0.2; //limits the LutVal change during the Luts update

    //for lut initialization
    float initSlopeMin = 0;
    float initSlopeMax = 0;

    float ditherRate = 0;

    //for SumNode
    float outValOffset = 0;
    unsigned int biasShift = 0;

    bool shiftLastGradient = false;

    //for building the 2D trees or nets
    size_t strideX = 1;
    size_t strideY = 1;

    size_t repeatX = 1;
    size_t repeatY = 1;

    size_t sizeX = 0;
    size_t sizeY = 0;

    size_t sizeTileX = 1;
    size_t sizeTileY = 1;
};

typedef std::unique_ptr<LayerConfig> LayerConfigPtr;

typedef std::vector<LayerConfigPtr> LayersConfigs;

} /* namespace lutNN */

#endif /* INTERFACE_LAYERCONFIG_H_ */
