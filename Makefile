all: train.f95
	flang-20 -g -O0 ./train.f95 -o trainflang
	gfortran -g -O0 ./train.f95 -o traingnu
