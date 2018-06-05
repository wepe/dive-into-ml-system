src=src/python_wrapper.cc src/lr.cc src/utils.cc
main:$(src)
	g++ -fPIC -shared -o liblr.so $(src)
