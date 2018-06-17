src=src/python_wrapper.cc src/lr.cc src/utils.cc
main:$(src)
	g++ -fPIC -shared -fopenmp -o liblr.so $(src)
