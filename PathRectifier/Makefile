PLATFORM = linux64
GUROBI_HOME = /opt/gurobi604/linux64
INC      = $(GUROBI_HOME)/include/
CPP      = g++
CARGS    = -m64 -g -std=c++14
CPPLIB   = -L $(GUROBI_HOME)/lib/ -lgurobi_c++ -lgurobi60
PROJ_DIR = /home/erik/Projects/c++



PathRectifier: $(PROJ_DIR)/PathRectifier/PathRectifier.cpp
	$(CPP) $(CARGS) -o $(PROJ_DIR)/PathRectifier/PathRectifier $(PROJ_DIR)/PathRectifier/PathRectifier.cpp -I$(INC) $(CPPLIB) -lpthread -lm