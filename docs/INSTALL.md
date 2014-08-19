HOW TO INSTALL

# Needed softwares
 * Lex, Yacc
 * C and Fortran Compilers (supports C99 and Fortran 90)
 * C++
 * Java Compiler
 * Apache Ant (1.8.1 or later)
 * MPI Implementation (supports MPI-2)
 * libxml2
 * make

## On Debian GNU/Linux and Ubuntu
 The following packages may be needed to be installed previously:

 * flex gcc gfortran g++ openjdk-7-jdk ant libopenmpi-dev openmpi-bin libxml2-dev byacc make

## On Red Hat and CentOS
 The following packages may be needed to be installed previously:

 * flex gcc gfortran gcc-c++ java-1.7.0-openjdk-devel ant openmpi-devel libxml2-devel byacc make

## If using the IBM Java Compiler
 Need to change import setting
 import com.sun.org.apache.xml.internal.serializer.OutputPropertiesFactory; ->  import org.apache.xml.serializer.OutputPropertiesFactory;
 in 
 * XcodeML-Common/src/xcodeml/util/XmUtil.java
 * XcodeML-Exc-Tools/src/exc/util/omompx.java
 * F-BackEnd/src/xcodeml/f/decompile/XfDecompileDomVisitor.java

## Usage of local-view operations (coarray, post/wait, lock/unlock)
 * Need to install GASNet (http://gasnet.lbl.gov) except for the K computer and FX10.
 * On the K computer or FX10, you can use local-view operations by using Fujitsu RDMA.

## Usage of OpenACC compiler
 * Need to install CUDA (https://developer.nvidia.com/cuda-zone).

# Install Step
## Configure
### On a general linux cluster
    $ ./configure --prefix=[INSTALLATION PATH]
         or
    $ ./configure CPP="pgcc -E" CC=gcc FC=gfortran  // To use a PGI compiler

 If you want to use Coarray functions
    $ ./configure --with-gasnet=[GASNet INSTALLATION PATH] --with-gasnet-conduit=[GASNet-Conduit]

 If you want to use OpenACC compiler
    $ ./configure --enable-openacc --with-cuda=[CUDA INSTALLATION PATH]

### On the K computer or FX10
    $ ./configure --target=Kcomputer-linux-gnu --prefix=[INSTALLATION PATH]
    $ ./configure --target=FX10-linux-gnu --prefix=[INSTALLATION PATH]

### On Cray machines
    $ ./configure --target=Cray-linux-gnu --prefix=[INSTALLATION PATH]

### On SX machines
    $ ./configure --target=sx --prefix=[INSTALLATION PATH]

### On BlueGene/Q
    First of all, you need to install openJDK (openjdk1.7.0-ppc-aix-port-linux-ppc64-b**.tar.bz2)
    from http://cr.openjdk.java.net/~simonis/ppc-aix-port/
    After that, please set PATH.
    $ ./configure --target=powerpc-ibm-none --prefix=[INSTALLATION PATH]

## Build
    $ make; make install

## Set PATH
    $ export PATH=[INSTALLATION PATH]/bin:$PATH

## Test (Optional)
    $ make tests
    $ make run-tests

 If you want to clean binaries of tests,
    $ make clean-tests

# Note
 In some reasons you failed to build the XMP, especially in the configuration stage,
 issuing autogen.sh on the top directory would be a solution.
 And if the build still failed even after issuing the autogen.sh,
 you should update your autotools (autoconf/automake/libtools) to the latest ones.
