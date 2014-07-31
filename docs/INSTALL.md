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

## On BlueGene/Q
 Need to install openJDK(openjdk1.7.0-ppc-aix-port-linux-ppc64-b**.tar.bz2) downloaded from the following URL:
 http://cr.openjdk.java.net/~simonis/ppc-aix-port/

## Usage of IBM Java Compiler
 Need to change import setting
 import com.sun.org.apache.xml.internal.serializer.OutputPropertiesFactory; ->  import org.apache.xml.serializer.OutputPropertiesFactory;
 in 
 * XcodeML-Common/src/xcodeml/util/XmUtil.java
 * XcodeML-Exc-Tools/src/exc/util/omompx.java
 * XcodeML-Exc-Tools/src/exc/openacc/ACCmain.java
 * F-BackEnd/src/xcodeml/f/decompile/XfDecompileDomVisitor.java

## Usage of local-view operations (coarray, post/wait, lock/unlock)
 * Need to install GASNet (http://gasnet.lbl.gov) except for the K computer and FX10.
 * On the K computer or FX10, you can use local-view operations by using Fujitsu RDMA.

## Usage of OpenACC compiler
 * Need to install CUDA (https://developer.nvidia.com/cuda-zone).

# Install Step
## Configure
### On a general linux cluster
    $ ./configure --prefix=[INSTALL-DIR]
         or
    $ ./configure --with-backend-cc=mpicc --prefix=[INSTALL-DIR]

 If you want to use Coarray functions
    $ ./configure --with-gasnet=[GASNet-INSTALL-DIR] --with-gasnet-conduit=[GASNet-Conduit] --prefix=[INSTALL-DIR]

 If you want to use OpenACC compiler
    $ ./configure --with-cuda=[CUDA-INSTALL-DIR] --enable-openacc --prefix=[INSTALL-DIR]

### On the K computer or FX10
    $ ./configure --target=Kcomputer-linux-gnu --prefix=[INSTALL-DIR]
    $ ./configure --target=FX10-linux-gnu --prefix=[INSTALL-DIR]

### On Cray machines
    $ ./configure --target=Cray-linux-gnu --prefix=[INSTALL-DIR]

### On SX machines
    $ ./configure --target=sx --prefix=[INSTALL-DIR]

## Build
    $ make; make install

## Set PATH
    $ export PATH=[INSTALL-DIR]/bin:$PATH

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

