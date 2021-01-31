build:
	g++ -c -fPIC ./src/build.cpp -O3
	mv build.o kermdl.o 
	g++ -shared -o libkermdl.so kermdl.o 
#	cp libkermdl.so ~/.lib/
