TARGET_EXEC ?= lutNN2

LIBS = -lboost_timer -lboost_system -lboost_serialization -lboost_chrono

BUILD_DIR ?= ./bin
SRC_DIRS ?= ./src 
TEST_DIR ?= ./test
INC_DIRS=../../

BINS = mnistLutInter2 mnistLutBinaryNN mnistBinaryLutNN5 mnistBinaryLutNN4 gmtLutInter1 gmtBinaryLutNNTest gmtBinaryLutNN1 omtfRegression1 omtfRegression2 omtfRegressionTest omtfRegressionTest2 omtfClassifier2 omtfRegressionTestFixedPoint # omtfClassifier1 omtfClassifierTest  lutNetworkFixedPointTest omtfClassifierTestFixedPoint omtfRegressionTest omtfRegressionTestFixedPoint
# mnistBinaryLutNN3 mnistBinaryLutNN2 mnistLutInter1 

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.cc -or -name *.s)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) 
#OBJS += $(BINS:%=$(BUILD_DIR)/test/%.cpp.o)
DEPS := $(OBJS:.o=.d)

#BOOST_DIR = /cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/boost/1.67.0-pafccj
BOOST_DIR = /cvmfs/cms.cern.ch/slc7_amd64_gcc10/external/boost/1.75.0

HLS_INCLUDE_DIR=/cvmfs/cms.cern.ch/slc7_amd64_gcc10/external/hls/2019.08-cms/include

#INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS)) -I $(BOOST_DIR)/include -I $(HLS_INCLUDE_DIR)

#CPPFLAGS ?= $(INC_FLAGS) -MMD -MP
CXXFLAGS =-c -O3 -g -Wall -fmessage-length=0 -std=c++17 `root-config --cflags` $(INC_FLAGS) -MMD -MP -Wno-int-in-bool-context -Wno-maybe-uninitialized #-w #-Wall -Wextra #

LDFLAGS=`root-config --glibs` -L $(BOOST_DIR)/lib #-Wl,--trace

.SECONDARY: $(BINS:%=$(BUILD_DIR)/test/%.cpp.o)

all: $(BINS) #;echo $(OBJS) #
	
# in $(OBJS) we have only the obj made from the .cpp files in the SRC_DIRS
# therefore we must add a prequisition to make the obj from the file in the test dir containing main
# this file should have the name as the target with .cpp, and the target will be one of those given in all
%: $(OBJS) $(BUILD_DIR)/test/%.cpp.o
	echo "target1: " $@
	echo "printing obj: " $(OBJS)
	$(CXX)  $(OBJS) $(BUILD_DIR)/test/$@.cpp.o -o $(BUILD_DIR)/$@.exe $(LDFLAGS) $(LIBS)

# assembly
$(BUILD_DIR)/%.s.o: %.s
	echo "target2: " $@ 
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) -c $< -o $@

# c source
#$(BUILD_DIR)/%.c.o: %.c
#	$(MKDIR_P) $(dir $@)
#	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	echo "target3: " $@ 
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p


	