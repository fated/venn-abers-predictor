CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: va-offline va-online va-cv

va-offline: va-offline.cpp utilities.o svm.o va.o
	$(CXX) $(CFLAGS) va-offline.cpp utilities.o svm.o va.o -o va-offline -lm

va-online: va-online.cpp utilities.o svm.o va.o
	$(CXX) $(CFLAGS) va-online.cpp utilities.o svm.o va.o -o va-online -lm

va-cv: va-cv.cpp utilities.o svm.o va.o
	$(CXX) $(CFLAGS) va-cv.cpp utilities.o svm.o va.o -o va-cv -lm

utilities.o: utilities.cpp utilities.h
	$(CXX) $(CFLAGS) -c utilities.cpp

svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp

va.o: va.cpp va.h
	$(CXX) $(CFLAGS) -c va.cpp

clean:
	rm -f utilities.o svm.o va.o va-offline va-online va-cv
