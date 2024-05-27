INC_DIR=-I../
#CC=g++
#CFLAGS=-c -g -Wall `root-config --cflags`
LDFLAGS=`root-config --glibs`

CXXFLAGS =-c -O3 -g -Wall -fmessage-length=0 -std=c++1y `root-config --cflags` $(INC_DIR)

OBJS =		src/Node.o src/LutNetwork.o src/Utlis.o src/MNISTParser.o src/InputNodeFactory.o src/mnist2.o  #src/lutNN.o src/mnistOneOutVal1.o # mnist_older

LIBS =

TARGET =	mnist2.exe #tf.exe #lutNN.exe 

$(TARGET):	$(OBJS)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
