
build:
	g++ -c ./src/build.cpp -lblas -llapack -O3
	mv build.o kermdl.o 

