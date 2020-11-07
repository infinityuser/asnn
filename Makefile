build:
	g++ -c ./src/build.cpp -O3
	mv build.o kermdl.o 
	g++ -shared -o libkermdl.so kermdl.o 
