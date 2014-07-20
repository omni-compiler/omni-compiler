README

# How to install
 Please show "docs/INSTALL.md"

-----
# Implementation Status
 Some features in the language specification are not supported in this release.
 (You can download the specification of XMP from "http://www.xcalablemp.org".)

 Please show "docs/STATUS/$(version).md"

-----
# How to use
## Compile
    $ xmpcc -O2 test.c -o test
    $ xmpf90 -O2 test.f90 -o test

## Compile (OpenACC)
    $ ompcc -acc -O2 test.c -o test

## Run
 Please use the command for running mpi program on your environment.
    $ mpirun -np 4 test

## Environment Variable
* XMP_COARRAY_HEAP_SIZE
 **Note that on the K computer this value is not used.**
 This value is used to malloc for coarray. Therefore, this value must be 
 set as a total coarray size. The default size is 16MB.
 If you want to change this value, please set it as below.

    export XMP_COARRAY_HEAP_SIZE=128M

* XMP_COARRAY_STRIDE_SIZE
 **Note that on the K computer this value is not used.**
 This value is used to malloc for coarray stride operations.
 The default size is 1MB.
 If you want to change this value, please set it as below.

    export XMP_COARRAY_STRIDE_SIZE=32M

# Profiling Options in XMP/C
 XMP supports profiler interfaces of Scalasca and tlog.
 The tlog is included in XMP.
 If you want to use the profiler interface of Scalasca,
 you need to download and compile it from Scalasca website.
 Moreover you need to set an environmental variable as below,

    $ export SCALASCA_HOME = [Scalasca-INSTALL-DIR]

 If you want to specify directives for profiling,
 please add "profile" to directive.

     #pragma xmp loop on t(i) profile

 When you want to compile XMP source code with profiling,
 please add some of below options.
    -profile         Emit XMP directive profiling code only for specified directives
    -allprofile      Emit XMP directive profiling code for all directives
    -with-scalasca   Emit Scalasca instrumentation
    -with-tlog       Emit tlog instrumentation
