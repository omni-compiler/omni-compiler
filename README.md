README
-----
# Manual
  See http://omni-compiler.org/manual.html

-----
# Implementation Status
 Some features in the language specification are not supported in this release.
  (You can download the specification of XMP from "http://xcalablemp.org")

 See docs/STATUS-XMP.md and docs/STATUS-CAF.md

----- 
# Quick Reference for XMP
## How to install
 $ ./configure --prefix=(INSTALL PATH)
 $ make; make install
 $ export PATH=(INSTALL PATH)/bin:$PATH

## Compile
 $ xmpcc  -O2 test.c   -o test
 $ xmpf90 -O2 test.f90 -o test

## Execute
$ mpirun -np 4 ./test

-----
# Quick	Reference for OpenACC
## How to install
 $ ./configure --prefix=(INSTALL PATH) --enable-openacc --with-cuda=(CUDA PATH) 
 $ make; make install
 $ export PATH=(INSTALL PATH)/bin:$PATH

## Compile
 $ ompcc -acc -O2 test.c -o test

## Execute
$ ./test
