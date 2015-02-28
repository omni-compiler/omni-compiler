AC_PREREQ(2.60)
LT_PREREQ(2.4)

AC_INIT([Omni Compiler], [0.9.1], [h-murai@riken.jp], [omni])

AC_CONFIG_AUX_DIR([./buildutils])
AC_CONFIG_MACRO_DIR([./buildutils])

AC_CANONICAL_TARGET

AC_CONFIG_SRCDIR([include/exc_platform.h])
AC_CONFIG_HEADER([include/config.h Driver/atool/src/include/nata/nata_config.h])

# --------------------------------------------------------------------
# check bash
SHELL=`which bash` 1> /dev/null 2> /dev/null
if test $? -ne 0; then
   AC_MSG_ERROR([To run libtool you need a bash, not /bin/sh.])
   exit 1
fi
export SHELL
AC_SUBST(SHELL)

# --------------------------------------------------------------------
# check existence of commands set by user
for cmd	in $CC $FC $MPI_CC $MPI_FC $JAVA $JAVAC $JAR
do
  if test "$cmd" != ""; then
    which $cmd 2> /dev/null 1> /dev/null
    if test $? -eq 1; then
      echo "$cmd not found"
      exit 1
    fi
  fi
done

# save environmental variables set by user
userCFLAGS=${CFLAGS}
userFCFLAGS=${FCFLAGS}

userMPI_CPP=${MPI_CPP}
userMPI_CPPFLAGS=${MPI_CPPFLAGS}
userMPI_CC=${MPI_CC}
userMPI_CFLAGS=${MPI_CFLAGS}
userMPI_CLIBS=${MPI_CLIBS}

userMPI_FPP=${MPI_FPP}
userMPI_FPPFLAGS=${MPI_FPPFLAGS}
userMPI_FC=${MPI_FC}
userMPI_FCFLAGS=${MPI_FCFLAGS}
userMPI_FCLIBS=${MPI_FCLIBS}

userJAVA=${JAVA}
userJAVAC=${JAVAC}
userJAR=${JAR}

# --------------------------------------------------------------------
# Initialize for autoconf
LT_INIT

AC_SUBST(LIBTOOL)
AC_SUBST(LIBTOOL_DEPS)
AM_INIT_AUTOMAKE([no-define foreign])
AM_MAINTAINER_MODE([enable])

TOPDIR=${ac_pwd}
AC_SUBST(TOPDIR)

# --------------------------------------------------------------------
# check project dirs
AC_MSG_CHECKING(check XcodeML-Common project exists)

if test -e "XcodeML-Common/Makefile.am"; then
    XCODEML_COMMON_DIR=XcodeML-Common
    AC_CONFIG_FILES($XCODEML_COMMON_DIR/Makefile)
    AC_MSG_RESULT(yes)
else
    XCODEML_COMMON_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(XCODEML_COMMON_DIR)

# --------------------------------------------------------------------
AC_MSG_CHECKING(check C-FrontEnd project exists)

if test -e "C-FrontEnd/src/Makefile.am"; then
    C_FRONTEND_DIR=C-FrontEnd/src
    AC_CONFIG_FILES($C_FRONTEND_DIR/Makefile)
    AC_MSG_RESULT(yes)
else
    C_FRONTEND_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(C_FRONTEND_DIR)

# --------------------------------------------------------------------
AC_MSG_CHECKING(check C-BackEnd project exists)

if test -e "C-BackEnd/Makefile.am"; then
    C_BACKEND_DIR=C-BackEnd
    AC_CONFIG_FILES($C_BACKEND_DIR/Makefile)
    AC_MSG_RESULT(yes)
else
    C_BACKEND_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(C_BACKEND_DIR)

# --------------------------------------------------------------------
AC_MSG_CHECKING(check F-FrontEnd project exists)

if test -e "F-FrontEnd/src/Makefile.am"; then
    F_FRONTEND_DIR=F-FrontEnd/src
    AC_CONFIG_FILES($F_FRONTEND_DIR/Makefile)
    AC_MSG_RESULT(yes)
else
    F_FRONTEND_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(F_FRONTEND_DIR)

# --------------------------------------------------------------------
AC_MSG_CHECKING(check F-BackEnd project exists)

if test -e "F-BackEnd/Makefile.am"; then
    F_BACKEND_DIR=F-BackEnd
    AC_CONFIG_FILES($F_BACKEND_DIR/Makefile)
    AC_MSG_RESULT(yes)
else
    F_BACKEND_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(F_BACKEND_DIR)

# --------------------------------------------------------------------
AC_MSG_CHECKING(check XcodeML-Exc-Tools project exists)

if test -e "XcodeML-Exc-Tools/Makefile.am"; then
    XCODEML_EXC_TOOLS_DIR=XcodeML-Exc-Tools
    AC_CONFIG_FILES($XCODEML_EXC_TOOLS_DIR/Makefile)
    AC_CONFIG_FILES($XCODEML_EXC_TOOLS_DIR/src/exc/util/MachineDepConst.java)
    AC_MSG_RESULT(yes)
else
    XCODEML_EXC_TOOLS_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(XCODEML_EXC_TOOLS_DIR)

# --------------------------------------------------------------------
AC_MSG_CHECKING(check Driver project exists)

if test -e "Driver/src/Makefile.am"; then
    OMDRIVER_DIR=Driver/src
    AC_CONFIG_FILES($OMDRIVER_DIR/Makefile)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/ompcc.conf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/ompf90.conf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/xmpcc.conf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../etc/xmpf90.conf)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/ompcc)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/ompf90)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/xmpcc)
    AC_CONFIG_FILES($OMDRIVER_DIR/../bin/xmpf90)
    AC_CONFIG_FILES($OMDRIVER_DIR/../libexec/omni_traverse)
    AC_MSG_RESULT(yes)
else
    OMDRIVER_DIR=''
    AC_MSG_RESULT(no)
fi

AC_SUBST(OMDRIVER_DIR)

# --------------------------------------------------------------------
# C
LIBXMP_HEADER_DIR=libxmp/include
LIBXMP_DIR=libxmp/src
LIBXMP_THREADS_DIR=libxmp/src_threads
LIBXMP_GPU_DIR=libxmp/src_gpu
AC_CONFIG_FILES($LIBXMP_HEADER_DIR/Makefile)
AC_CONFIG_FILES($LIBXMP_DIR/Makefile)
AC_CONFIG_FILES($LIBXMP_THREADS_DIR/Makefile)
AC_CONFIG_FILES($LIBXMP_GPU_DIR/Makefile)
AC_SUBST(LIBXMP_HEADER_DIR)
AC_SUBST(LIBXMP_DIR)
AC_SUBST(LIBXMP_THREADS_DIR)
AC_SUBST(LIBXMP_GPU_DIR)

# F
LIBXMPF_HEADER_DIR=libxmpf/include
LIBXMPF_DIR=libxmpf/src
AC_CONFIG_FILES($LIBXMPF_HEADER_DIR/Makefile)
AC_CONFIG_FILES($LIBXMPF_DIR/Makefile)
AC_SUBST(LIBXMPF_HEADER_DIR)
AC_SUBST(LIBXMPF_DIR)

# --------------------------------------------------------------------
# check on what system we are running.

#AC_CANONICAL_SYSTEM
AC_MSG_CHECKING(os)

HOSTARCH=`./buildutils/config.guess | grep -v Linking`
CPU=`echo $HOSTARCH | awk -F- '{ print $1 }' | sed 's/\./_/g'`
OS=`echo $HOSTARCH | awk -F- '{ print $3 }' | sed 's/\./_/g'`
PPTGT=""
MM5TGT=""

case $CPU in
    i*86)
        CPU=i386;;
    x86_64*)
        CPU=x86_64;;
    alpha*)
        CPU=alpha;;
esac

case $OS in
    sunos*)
        OS=sunos;;
    solaris*)
        OS=solaris;;
    freebsd*)
        OS=freebsd;;
    netbsd*)
        OS=netbsd;;
    irix*)
        OS=irix;;
    aix*)
        OS=aix;;
    darwin*)
        OS=darwin;;
    cygwin*)
        OS=cygwin32;;
esac

case $HOSTARCH in
    i*86-pc-linux*)
        PPTGT=linux;;
    alpha*-unknown-linux*)
        PPTGT=linux;;
    i386-pc-solaris*)
        PPTGT=sol2;;
    sparc-sun-solaris*)
        PPTGT=sol2;;
    mips-sgi-irix*)
        PPTGT=orig2k;;
esac

AC_MSG_RESULT($OS $CPU)
AC_SUBST(OS)
AC_SUBST(CPU)
AC_SUBST(MM5TGT)
AC_SUBST(PPTGT)

# --------------------------------------------------------------------
# OpenMP runtime checking

def_thread_type=""
def_lock_type=""
available_threads=""
available_locks=""
ignore_pair=""
ompcLibAsmSrcs=""

# Default thread type
# Check whether --with-thread or --without-thread was given.
AC_ARG_WITH(thread,
    [  --with-thread               type of openmp thread],
    [thread_type=$withval], [thread_type=""])

# Default lock type
# Check whether --with-lock or --without-lock was given.
AC_ARG_WITH(lock,
    [  --with-lock                 type of openmp lock],
    [lock_type=$withval], [lock_type=""])

case $OS in
    solaris*)
	def_thread_type=solaris
	def_lock_type=spin
	available_locks="mutex spin"
	available_threads="solaris pthread"
	case $CPU in
	    sparc*)
		ompcLibAsmSrcs="asm_sparc.s";;
	esac
	;;
    irix*)
	def_thread_type=sproc
	def_lock_type=spin
	available_locks="mutex spin"
	available_threads="pthread sproc"
	ignore_pair="sproc mutex"
	;;
    *)
	def_thread_type=pthread
	def_lock_type=mutex
	available_locks="mutex"
	available_threads="pthread"
	case $CPU in
	    i386*|mips*)
		def_lock_type=spin
		available_locks="mutex spin";;
	    sparc*)
		def_lock_type=spin
		available_locks="mutex spin"
		ompcLibAsmSrcs="asm_sparc.s";;
	    alpha*)
		def_lock_type=spin
		available_locks="mutex spin"
		ompcLibAsmSrcs="asm_alpha.s";;
	    rs6000*)
		available_locks="mutex spin";;
	    *)
		true;;
	esac
	;;
esac

if test "X${thread_type}" = "X"; then
    thread_type=${def_thread_type}
fi

if test "X${lock_type}" = "X"; then
    lock_type=${def_lock_type}
fi

OMP_THREAD_TYPE=${thread_type}
AC_SUBST(OMP_THREAD_TYPE)
OMP_LOCK_TYPE=${lock_type}
AC_SUBST(OMP_LOCK_TYPE)

case $lock_type in
  spin*)
    case $CPU in i386*|x86_64*|mips*|sparc*|alpha*|rs6000*)
      true;;
      *)
      echo "$ac_t""spin lock is only available on i386, x86_64, mips, alpha, sparc and rs6000 architectures." 1>&6
      exit 1;;
    esac;;
  *)
    true;;
esac

THREAD_DEF=""
case $thread_type in
    solaris*)
	THREAD_DEF="-DUSE_SOL_THREAD"
	;;
    sproc*)
	THREAD_DEF="-DUSE_SPROC"
	;;
    *)
	THREAD_DEF="-DUSE_PTHREAD"
	;;
esac

LOCK_DEF=""
case $lock_type in
    mutex*)
	LOCK_DEF=""
	;;
    spin*)
	LOCK_DEF="-DUSE_SPIN_LOCK"
	;;
esac

OMPC_LIB_ASM=${ompcLibAsmSrcs}

# --------------------------------------------------------------------
genLibTargetTmpl () {
    dir=$1
    tmpl=${dir}/target.tmpl
    echo "$ac_t""creating ${tmpl}" 1>&6
    > ${tmpl}
    ./buildutils/genLibTarget --locks="${available_locks}" --threads="${available_threads}" \
	--def_lock=${lock_type} --def_thread=${thread_type} --ignore_combo="${ignore_pair}" \
	--os=${OS} --cpu=${CPU} \
	`(cd ${dir};grep --regexp="^SRCS" Makefile.am | sed s/"SRCS		= "//g)` > ${tmpl}
}

# --------------------------------------------------------------------
LIBOMPC_DIR="libompc/src"
AC_CONFIG_FILES($LIBOMPC_DIR/Makefile)
LIBOMPF_DIR="libompf/src"
AC_CONFIG_FILES($LIBOMPF_DIR/Makefile)
LIBTLOG_DIR="libtlog/src_threads"
AC_CONFIG_FILES($LIBTLOG_DIR/Makefile)
LIBTLOG_MPI_DIR="libtlog/src_mpi"
AC_CONFIG_FILES($LIBTLOG_MPI_DIR/Makefile)
AC_CONFIG_FILES($LIBTLOG_MPI_DIR/bin/tlogview)
genLibTargetTmpl $LIBOMPC_DIR

AC_SUBST(LIBOMPC_DIR)
AC_SUBST(LIBOMPF_DIR)
AC_SUBST(LIBTLOG_DIR)
AC_SUBST(LIBTLOG_MPI_DIR)

# --------------------------------------------------------------------
LIBACC_DIR=libacc/src
LIBACC_HEADER_DIR=libacc/include
AC_CONFIG_FILES($LIBACC_DIR/Makefile)
AC_CONFIG_FILES($LIBACC_HEADER_DIR/Makefile)
AC_SUBST(LIBACC_DIR)
AC_SUBST(LIBACC_HEADER_DIR)

# --------------------------------------------------------------------
AC_ARG_ENABLE(debug,
    [  --enable-debug                 enable generate executable with debug symbol (false default)],
    [doDebug=yes], [doDebug=no])

AC_ARG_ENABLE(openacc,
    [  --enable-openacc               enable OpenACC (false default)],
    [doOpenAcc=yes], [doOpenAcc=no])

AC_ARG_ENABLE(atool,
    [  --enable-atool                 enable atool (false default)],
    [doatool=yes], [doatool=no])

AC_ARG_ENABLE(xacc,
    [  --enable-xacc                  enable XcalableACC (false default)],
    [xacc=yes], [xacc=no])

## Add temporarily by mnakao
#AC_ARG_ENABLE(FJRDMA,
#   [  --enable-FJRDMA                enable RDMA functions on the K compiter or FX10],
#   [enableFJRDMA=yes], [enableFJRDMA=no])
#
#AM_CONDITIONAL(FJRDMA, test x${enableFJRDMA} = xyes)
## End Add temporarily

AM_CONDITIONAL(FJRDMA, test "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu")

# --------------------------------------------------------------------
# check basic commands

AC_PROG_CPP
AC_PROG_CC
AC_PROG_FC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_AWK
AC_PROG_LEX
AC_PROG_YACC

# --------------------------------------------------------------------
# set XMP-dev

#XMP_ENABLE_GPU=0
#if test "x${doXmpGpu}" = "xyes"; then
#    XMP_ENABLE_GPU=1
#fi
#AM_CONDITIONAL(ENABLE_XMP_GPU, test "x${doXmpGpu}" = "xyes")
#AC_SUBST(XMP_ENABLE_GPU)

# --------------------------------------------------------------------
# set OpenACC
ENABLE_ACC=0
if test "x${doOpenAcc}" = "xyes"; then
    ENABLE_ACC=1
fi
AM_CONDITIONAL(ENABLE_ACC, test "x${doOpenAcc}" = "xyes")
AC_SUBST(ENABLE_ACC)

# --------------------------------------------------------------------
# set ATOOLS
ENABLE_ATOOL=0
if test "x${doatool}" = "xyes"; then
   ENABLE_ATOOL=1
fi
AM_CONDITIONAL(ENABLE_ATOOL, test "x${doatool}" = "xyes")
AC_SUBST(ENABLE_ATOOL)

# --------------------------------------------------------------------
# set XACC
AM_CONDITIONAL(ENABLE_XACC, test "x${xacc}" = "xyes")
AC_SUBST(ENABLE_XACC)

# --------------------------------------------------------------------
# Generate cpu/os specifier
CPUDEF=-DOMNI_CPU_`echo $CPU | tr '[a-z]' '[A-Z]'`
OSDEF=-DOMNI_OS_`echo $OS | tr '[a-z]' '[A-Z]'`

if test "$target" != "$host"; then
   TARGET_CPU=`echo $target | awk -F- '{ print $1 }' | sed 's/\./_/g'`
   TARGET_CPUDEF=-DOMNI_TARGET_CPU_`echo ${TARGET_CPU} | tr '[a-z]' '[A-Z]'`
fi

debCppFlags=""
if test "x${doDebug}" = "xyes"; then
    CFLAGS="-g -O0"
    FCFLAGS="-g -O0"
    debCppFlags="-DDEBUG"
else
    debCppFlags="-DNDEBUG"
fi

CPPFLAGS="${CPUDEF} ${OSDEF} ${debCppFlags} ${TARGET_CPUDEF}"
AC_SUBST(CPPFLAGS)

CXXFLAGS=$CFLAGS
AC_SUBST(CXXFLAGS)

AH_TEMPLATE(OBJEXT, [object file extension])
AC_DEFINE_UNQUOTED(OBJEXT, "${OBJEXT}")

# --------------------------------------------------------------------
# Set compilers and their options:
#   The following operation sets 3 kinds of compiler and their options.
#
# (1) Local compiler (for compiling C_Front and so on)
#   CC, FC, CFLAGS, FCFLAGS, ...
#
# (2) MPI compiler (for compiling libxmp and so on)
#  MPI_CC, MPI_FC, MPI_CFLAGS, MPI_FCFLAGS, ...
#
# (3) Omni compiler (for using xmpcc and xmpf90 commands)
#  OMNI_XXX_CFLAGS, OMNI_XXX_FCFLAGS, ...
#    "XXX" is a phase name.
#
# Note that:
#  MPI_CFLAGS needs options for OpenMP and strong optimization (e.g. -O2) and C99.
#  MPI_FCFLAGS needs an option for strong optimization (e.g. -O2).
#
#  Prefix "OMNI_" options should set a minimal of option as possible.
#  OMNI_C_PP, Pre-processor command, needs an option for C99.
#  OMNI_NTV_CFLAGS needs options for C99 support.
#  OMNI_LNK_CFLAGS and OMNI_LNK_FCFLAGS need an option for OpenMP.

# Memo:
#   Name rule for --target=machine-[vendor-]operatingsystem
#   It contains three fields: the name of the CPU family/model, the vendor, and the operating system name.
#   But, the vendor field is mostly unused.
#   Refer: http://wiki.osdev.org/Target_Triplet

if test "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu"; then
# OpenMP
  OPENMP_OPT="-Kopenmp"

# XcalableMP
  XMPC_DEF="-D_XCALABLEMP"
  XMPF_DEF="-D_XCALABLEMP"

# MPI C
  MPI_CC="mpifccpx"
  MPI_CPP=$MPI_CC
  MPI_CFLAGS="-Kfast,openmp -Xg -noansi"
  MPI_CPPFLAGS="-E -Xg -noansi"
  MPI_CLIBS=""

# MPI F
  MPI_FC="mpifrtpx"
  MPI_FPP=$MPI_FC
  MPI_FPPFLAGS="-E"
  MPI_FCFLAGS="-Kfast,openmp"
  MPI_MODINC="-M"
  MPI_FCLIBS=""

# Omni C
  OMNI_C2X_FLAGS=""
  OMNI_NTV_CFLAGS="-Xg -noansi"
  OMNI_LNK_CFLAGS="$OMNI_NTV_CFLAGS $OPENMP_OPT"

# Omni F
  OMNI_F2X_FLAGS=""
  OMNI_NTV_FCFLAGS="-Free"
  OMNI_LNK_FCFLAGS="$OMNI_NTV_FCFLAGS $OPENMP_OPT"

# Omni Common
  OMNI_X2X_FLAGS=""

elif test "$target" = "MIC-linux-gnu"; then
# OpenMP
  OPENMP_OPT="-mp"

# XcalableMP
  XMPC_DEF="-D_XCALABLEMP"
  XMPF_DEF="-D_XCALABLEMP"

# MPI C
  MPI_CC="mpiicc"
  MPI_CPP=$MPI_CC
  MPI_CPPFLAGS="-E -std=gnu99"
  MPI_CFLAGS="-fast $OPENMP_OPT"
  MPI_CLIBS=""

# MPI F
  MPI_FC="mpiifort"
  MPI_FPP=$MPI_FC
  MPI_FPPFLAGS="-E"
  MPI_FCFLAGS="-fast $OPENMP_OPT"
  MPI_MODINC="-module"
  MPI_FCLIBS=""

# Omni C
  OMNI_C2X_FLAGS=""
  OMNI_NTV_CFLAGS="-std=gnu99"
  OMNI_LNK_CFLAGS="$OMNI_NTV_CFLAGS $OPENMP_OPT"

# Omni F
  OMNI_F2X_FLAGS=""
  OMNI_NTV_FCFLAGS=""
  OMNI_LNK_FCFLAGS="$OMNI_NTV_FCFLAGS $OPENMP_OPT"

# Omni Common
  OMNI_X2X_FLAGS=""

elif test "$target" = "Cray-linux-gnu"; then
# OpenMP
  OPENMP_OPT="-h omp"

# XcalableMP
  XMPC_DEF="-D_XCALABLEMP"
  XMPF_DEF="-D_XCALABLEMP"

# MPI C
  MPI_CC="cc"
  MPI_CPP=$MPI_CC
  MPI_CPPFLAGS="-E"
  MPI_CFLAGS="-O2 $OPENMP_OPT"
  MPI_CLIBS=""

# MPI F
  MPI_FC="ftn"
  MPI_FPP=$MPI_FC
  MPI_FPPFLAGS="-E"
  MPI_FCFLAGS="-O2 $OPENMP_OPT"
  MPI_MODINC="-J"
  MPI_FCLIBS=""

# Omni C
  OMNI_C2X_FLAGS=""
  OMNI_NTV_CFLAGS="-hgnu -D_XMP_CRAY"
  OMNI_LNK_CFLAGS="$OMNI_NTV_CFLAGS $OPENMP_OPT"

# Omni F
  OMNI_F2X_FLAGS=""
  OMNI_NTV_FCFLAGS="-f free"
  OMNI_LNK_FCFLAGS="$OMNI_NTV_FCFLAGS $OPENMP_OPT"

# Omni Common
  OMNI_X2X_FLAGS=""

elif test "$target" = "sx-nec-superux"; then
# OpenMP
  OPENMP_OPT="-Popenmp"

# XcalableMP
  XMPC_DEF="-D_XCALABLEMP"
  XMPF_DEF="-D_XCALABLEMP"

# MPI C
  MPI_CC="sxmpic++"
  MPI_CPP=$MPI_CC
  MPI_CPPFLAGS="-E"
  MPI_CFLAGS="-O2 $OPENMP_OPT"
  MPI_CLIBS=""

# MPI F
  MPI_FC="sxmpif90"
  MPI_FPP=$MPI_FC
  MPI_FPPFLAGS="-E"
  MPI_FCFLAGS="-O2 $OPENMP_OPT"
  MPI_MODINC="-to"
  MPI_FCLIBS=""

# Omni C
  OMNI_C2X_FLAGS=""
  OMNI_NTV_CFLAGS="-Kgcc -w"
  OMNI_LNK_CFLAGS="$OMNI_NTV_CFLAGS $OPENMP_OPT"

# Omni F
  OMNI_F2X_FLAGS=""
  OMNI_NTV_FCFLAGS="-f4 -w"
  OMNI_LNK_FCFLAGS="$OMNI_NTV_FCFLAGS $OPENMP_OPT"

# Omni Common
  OMNI_X2X_FLAGS=""

elif test "$target" = "powerpc-ibm-cnk"; then
  # These options are referred to https://scwwwb.kek.jp/index.html

# OpenMP
  OPENMP_OPT="-qsmp=omp"

# XcalableMP
  XMPC_DEF="-D_XCALABLEMP"
  XMPF_DEF="-WF,-D_XCALABLEMP"

# MPI C
  MPI_DIR="/bgsys/drivers/V1R2M1/ppc64/comm/bin/xl/"
  if test -d $MPI_DIR; then
    MPI_CC="$MPI_DIR/mpixlcxx_r"
  else
    MPI_CC="mpixlcxx_r"
  fi

# $MPICC cannot be used for XcodeML bug. 
# Instead of $MPICC, $CC (gcc) is used. 
# But, this is not a good idea. Wish to fix the bug.
#  MPI_CPP=$MPI_CC
  MPI_CPP=$CC
  MPI_CPPFLAGS="-E --std=c99"

  if test -d "/bgsys/drivers/ppcfloor/comm/include"; then
    MPI_CPPFLAGS="$MPI_CPPFLAGS -I/bgsys/drivers/ppcfloor/comm/include"
  fi

  MPI_CFLAGS="-O3 -qhot -qsimd=auto -q64 -qsource -qlist $INC_PATH $OPENMP_OPT"

  if test -f /usr/local/bg/lib/libkek.a; then
    MPI_CLIBS="/usr/local/bg/lib/libkek.a"
  fi

  if test -f /bgsys/drivers/ppcfloor/bgpm/lib/libbgpm.a; then
    MPI_CLIBS="$MPI_CLIBS /bgsys/drivers/ppcfloor/bgpm/lib/libbgpm.a"
  fi
  
# MPI F
  if test -d $MPI_DIR; then
    MPI_FC="/bgsys/drivers/V1R2M1/ppc64/comm/bin/xl/mpixlf90_r"
  else
    MPI_FC="mpixlf90_r"
  fi
  MPI_FPP=$MPI_FC
  MPI_FPPFLAGS="-E"
  MPI_FCFLAGS="-O3 -qhot -qsimd=auto -q64 -qsource -qlist -qextname $INC_PATH $OPENMP_OPT"
  MPI_MODINC="-I"

  INC_PATH="/usr/local/bg/include"
  if test -d $INC_PATH; then
    MPI_CFLAGS="$MPI_CFLAGS -I$INC_PATH"
    MPI_FCFLAGS="$MPI_FCFLAGS -I$INC_PATH"
  fi

  MPI_FCLIBS=$MPI_CLIBS

# Omni C
  OMNI_C2X_FLAGS=""
  OMNI_NTV_CFLAGS="-q64 -qipa"
  OMNI_LNK_CFLAGS="$OMNI_NTV_CFLAGS $OPENMP_OPT $MPI_CLIBS"

# Omni F
  OMNI_F2X_FLAGS=""
  OMNI_NTV_FCFLAGS="-q64 -qipa -qextname"
  OMNI_LNK_FCFLAGS="$OMNI_NTV_FCFLAGS $OPENMP_OPT $MPI_CLIBS"

# Omni Common
  OMNI_X2X_FLAGS=""

elif test "$target" = "powerpc-hitachi-aix"; then
# OpenMP
  OPENMP_OPT="-qsmp=omp"

# XcalableMP
  XMPC_DEF="-D_XCALABLEMP"
  XMPF_DEF="-D_XCALABLEMP"

# MPI C
  MPI_CC="mpCC_r"
  MPI_CPP=$MPI_CC
  MPI_CPPFLAGS="-E"
  MPI_CFLAGS="-q64 -qhot -qsimd=auto -O5 $OPENMP_OPT"
  MPI_CLIBS=""

# MPI F
  MPI_FC="mpxlf90_r"
  MPI_FPP=$MPI_FC
  MPI_FPPFLAGS="-E"
  MPI_FCFLAGS="-q64 -qhot -qsimd=auto -O5 -qextname $OPENMP_OPT"
  MPI_MODINC="-I"
  MPI_FCLIBS=""

# Omni C
  OMNI_C2X_FLAGS=""
  OMNI_NTV_CFLAGS="-q64"
  OMNI_LNK_CFLAGS="$OMNI_NTV_CFLAGS $OPENMP_OPT"

# Omni F
  OMNI_F2X_FLAGS=""
  OMNI_NTV_FCFLAGS="-q64 -qextname"
  OMNI_LNK_FCFLAGS="$OMNI_NTV_FCFLAGS $OPENMP_OPT"

# Omni Common
  OMNI_X2X_FLAGS=""

else
  # XcalableMP
  XMPC_DEF="-D_XCALABLEMP"
  XMPF_DEF="-D_XCALABLEMP"

  INTEL_MPICC=mpiicc
  which ${INTEL_MPICC} 1>/dev/null 2>/dev/null
  if test $? -eq 0; then
    MPI_CC=${INTEL_MPICC}
  else
    MPI_CC="mpicc"
  fi

  chk=`${MPI_CC} --version | head -n1`
  case $chk in
  cc* | gcc* | Apple*)
    # OpenMP
    OPENMP_OPT="-fopenmp"

    # XcalableMP
    XMPC_DEF="-D_XCALABLEMP"
    XMPF_DEF="-D_XCALABLEMP"

    # MPI C
    MPI_CPP=$MPI_CC
    MPI_CPPFLAGS="-E -std=gnu99"
    MPI_CFLAGS="-O2 -std=gnu99 -Wall -Wpointer-arith $OPENMP_OPT"
    MPI_CLIBS=""

    # MPI F
    MPI_FC="mpif90"
    MPI_FPP=$MPI_FC
    MPI_FPPFLAGS="-E"
    MPI_FCFLAGS="-O2 $OPENMP_OPT"
    MPI_MODINC="-J"
    MPI_FCLIBS=""

    # Omni C
    OMNI_C2X_FLAGS=""
    OMNI_NTV_CFLAGS="-std=gnu99"
    OMNI_LNK_CFLAGS="$OMNI_NTV_CFLAGS $OPENMP_OPT"

    # Omni F
    OMNI_F2X_FLAGS=""
    OMNI_NTV_FCFLAGS="-ffree-form -fcray-pointer"
    OMNI_LNK_FCFLAGS="$OMNI_NTV_FCFLAGS -fopenmp -lgomp"

    # Omni Common
    OMNI_X2X_FLAGS="-gnu"
    ;;
  *ICC*)
    # OpenMP
    OPENMP_OPT="-openmp"

    # XcalableMP
    XMPC_DEF="-D_XCALABLEMP"
    XMPF_DEF="-D_XCALABLEMP"

    # MPI C
    MPI_CPP=$MPI_CC
    ver=`$MPI_CC --version 2> /dev/null | head -n1`
    case $ver in
    *15*)
      MPI_CPPFLAGS="-E -std=gnu99 -no-gcc" ;;
    *)
      MPI_CPPFLAGS="-E -std=gnu99" ;;
    esac
    MPI_CFLAGS="-O2 -std=gnu99 -Wall -Wpointer-arith $OPENMP_OPT"
    MPI_CLIBS=""

    # MPI F
    MPI_FC="mpiifort"
    MPI_FPP=$MPI_FC
    MPI_FPPFLAGS="-E"
    MPI_FCFLAGS="-O2 $OPENMP_OPT"
    MPI_MODINC="-module"
    MPI_FCLIBS=""

    # Omni C
    OMNI_C2X_FLAGS=""
    OMNI_NTV_CFLAGS="-std=gnu99"
    OMNI_LNK_CFLAGS="$OMNI_NTV_CFLAGS $OPENMP_OPT"

    # Omni F
    OMNI_F2X_FLAGS=""
    OMNI_NTV_FCFLAGS=""
    OMNI_LNK_FCFLAGS="$OMNI_NTV_FCFLAGS $OPENMP_OPT"

    # Omni Common
    OMNI_X2X_FLAGS="-intel"
    ;;
  *)
    chk2=`$MPI_CC --version 2> /dev/null | head -n2 | tail -n1`
    case $chk2 in
    pgcc*)
      # OpenMP
      OPENMP_OPT="-mp"

      # XcalableMP
      XMPC_DEF="-D_XCALABLEMP"
      XMPF_DEF="-D_XCALABLEMP"

      # MPI C
      MPI_CPP=$MPI_CC
      MPI_CPPFLAGS="-E -c99"
      MPI_CFLAGS="-fast -c99 $OPENMP_OPT"
      MPI_CLIBS=""

      # MPI F
      MPI_FC="mpif90"
      MPI_FPP=$MPI_FC
      MPI_FPPFLAGS="-E"
      MPI_FCFLAGS="-fast $OPENMP_OPT"
      MPI_MODINC="-module"
      MPI_FCLIBS=""

      # Omni C
      OMNI_C2X_FLAGS="--no-builtin-va-arg"
      OMNI_NTV_CFLAGS="-D_XMP_PGI"
      OMNI_LNK_CFLAGS="$OMNI_NTV_CFLAGS $OPENMP_OPT"

      # Omni F
      OMNI_F2X_FLAGS=""
      OMNI_NTV_FCFLAGS=""
      OMNI_LNK_FCFLAGS="$OMNI_NTV_FCFLAGS $OPENMP_OPT"

      # Omni Common
      OMNI_X2X_FLAGS=""

      # Special options
      IS_PGI="Yes"
      AC_SUBST(IS_PGI)
      ;;
    *)
      echo "No supported compiler was found."
      echo "This program supports following MPI compilers."
      echo " - GNU Compiler (mpicc/mpif90)"
      echo " - Intel Compiler (mpiicc/mpiifort)"
      echo " - PGI Compiler (mpicc/mpif90)"
      echo " - Fujitsu Compiler (mpifccpx/mpifrtpx)"
      echo " - Cray Compiler (cc/ftn)"
      echo " - NEC Compiler (sxmpic++/sxmpif90)"
      echo " - IBM Compiler (mpixlcxx_r/mpixlf90_r)"
      echo " - HITACHI Compiler (mpCC_r/mpif90)"
      exit 1 ;;
    esac ;;
  esac
fi
AM_CONDITIONAL(POWERPC, test "$target" = "powerpc-ibm-cnk" -o "$target" = "powerpc-hitachi-aix")

# --------------------------------------------------------------------
# specify GPU C compiler

AC_ARG_WITH(gpucc,
    [  --with-gpucc=CMD            GPU C compiler (default: nvcc)],
    [GPU_CMD="${withval}"], [GPU_CMD="nvcc"])

AC_SUBST(GPU_CMD)

# --------------------------------------------------------------------
# specify GPU C compiler flags

AC_ARG_WITH(gpu-cflags,
    [  --with-gpu-cflags=CFLAGS    GPU C compiler flags (default: -arch=sm_20 -O3)],
    [GPU_CFLAGS="${withval}"], [GPU_CFLAGS="-arch=sm_20 -O3"])

AC_SUBST(GPU_CFLAGS)

# --------------------------------------------------------------------
# specify libgpu path

AC_ARG_WITH(cuda,
    [  --with-cuda=DIR             CUDA installation path (default: /usr/local/cuda)],
    [GPGPU_PREFIX="${withval}"], [GPGPU_PREFIX="/usr/local/cuda"])

AC_SUBST(GPGPU_PREFIX)

# --------------------------------------------------------------------
# specify libxmp path

AC_ARG_WITH(libxml,
    [  --with-libxml=DIR           libxml installation path (default: /usr)],
    [LIBXML_PREFIX="${withval}"], [LIBXML_PREFIX="/usr"])

if ! test -f $LIBXML_PREFIX/include/libxml2/libxml/parser.h; then
   AC_MSG_ERROR([parser.h not found. Please attach or check --with-libxml=<DIR>. See ./configure --help])
fi

AC_SUBST(LIBXML_PREFIX)
# --------------------------------------------------------------------
# set BLAS

AC_ARG_WITH(libblas,
    [  --with-libblas=DIR          BLAS Library (e.g. /foo/blas_LINUX.a)],
    [LIBBLAS="${withval}"; use_libblas=yes], [use_libblas=no])

if test x$use_libblas = xyes; then
   if ! test -f $LIBBLAS; then
     AC_MSG_ERROR([$LIBBLAS not found.])
   fi
   OMNI_LNK_CFLAGS="$OMNI_LNK_CFLAGS $LIBBLAS"
   OMNI_LNK_FCFLAGS="$OMNI_LNK_FCFLAGS $LIBBLAS"
else
  if test "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu"; then
    LIBBLAS="-SSL2BLAMP"
    OMNI_LNK_CFLAGS="$OMNI_LNK_CFLAGS $LIBBLAS"
    OMNI_LNK_FCFLAGS="$OMNI_LNK_FCFLAGS $LIBBLAS"
  fi
fi

AM_CONDITIONAL(USER_LIBBLAS, test x${use_libblas} = xyes)

# --------------------------------------------------------------------
# set TCA

AC_ARG_WITH(TCA,
    [  --with-TCA=DIR              TCA installation path],
    [TCA_PREFIX="${withval}"; use_tca=yes], [use_tca=no])

if test x$use_tca = xyes; then
   if ! test -f $TCA_PREFIX/tca-api.h; then
      AC_MSG_ERROR([$TCA_PREFIX/tca-api.h not found. Please check --with-TCA=<DIR>])
   fi

  MPI_CPPFLAGS="${MPI_CPPFLAGS} -I${TCA_PREFIX}"
  MPI_CFLAGS="${MPI_CFLAGS} -I${TCA_PREFIX}"
  MPI_CLIBS="${MPI_CLIBS} -L${TCA_PREFIX} -ltca"
  OMNI_LNK_CFLAGS="$OMNI_LNK_CFLAGS ${MPI_CLIBS}"
fi

AM_CONDITIONAL(TCA, test x${use_tca} = xyes)
# --------------------------------------------------------------------
# specify GASNet path

AC_ARG_WITH(gasnet,
    [  --with-gasnet=DIR           GASNet installation path],
    [GASNET_PREFIX="${withval}"; use_gasnet=yes], [use_gasnet=no])

if test x$use_gasnet = xyes; then
  if ! test -f "${GASNET_PREFIX}/include/gasnet.h"; then
    AC_MSG_ERROR([${GASNET_PREFIX}/include/gasnet.h not found. Please check --with-gasnet])
  fi
fi

AM_CONDITIONAL(GASNET, test x${use_gasnet} = xyes)

# --------------------------------------------------------------------
# specify GASNet conduit

AC_ARG_WITH(gasnet_conduit,
    [  --with-gasnet-conduit=NAME  GASNet conduit],
    [GASNET_CONDUIT="${withval}"], [GASNET_CONDUIT="${default_gasnet_conduit}"])

if test x${use_gasnet} = xyes; then
  gasnet_conduit_flag=no
  if test "${GASNET_CONDUIT}" = ""; then
    for c in smp udp mpi gm lapi elan vapi ibv portals gemini shmem dcmf sci
    do
      if test -d ${GASNET_PREFIX}/include/${c}-conduit; then
         GASNET_CONDUIT=${c}
         gasnet_conduit_flag=yes
      fi
    done
  else
    gasnet_conduit_flag=yes
  fi

  if test "${gasnet_conduit_flag}" = no || ! test -d $GASNET_PREFIX/include/${GASNET_CONDUIT}-conduit; then
    AC_MSG_ERROR([${GASNET_PREFIX}/include/${GASNET_CONDUIT}-conduit not found. Please attach or check --with-gasnet-conduit])
  else
    # Create temporary Makefile to set $(GASNET_CPPFLAGS) $(GASNET_CFLAGS) $(GASNET_LDFLAGS) $(GASNET_LIBS)
    __ECHO=""
    if test $OS = "darwin"; then
        __ECHO="echo"
    else
        __ECHO="echo -e"
    fi
    tmp_file=_XMP_makefile.$$
    mak_file=${GASNET_PREFIX}/include/${GASNET_CONDUIT}-conduit/${GASNET_CONDUIT}-seq.mak

    $__ECHO "include ${mak_file}\nall:\n\t@echo \$(GASNET_CPPFLAGS)" > ${tmp_file}
    GASNET_CPPFLAGS=`make -s -f ${tmp_file} 2>/dev/null`

    $__ECHO "include ${mak_file}\nall:\n\t@echo \$(GASNET_CFLAGS)"   > ${tmp_file}
    GASNET_CFLAGS=`make -s -f ${tmp_file} 2>/dev/null`

    $__ECHO "include ${mak_file}\nall:\n\t@echo \$(GASNET_LDFLAGS)"  > ${tmp_file}
    GASNET_LDFLAGS=`make -s -f ${tmp_file} 2>/dev/null`

    $__ECHO "include ${mak_file}\nall:\n\t@echo \$(GASNET_LIBS)"     > ${tmp_file}
    GASNET_LIBS=`make -s -f ${tmp_file} 2>/dev/null`
    rm ${tmp_file}
    unset __ECHO
  fi

  MPI_CPPFLAGS="${MPI_CPPFLAGS} ${GASNET_CPPFLAGS}"
  MPI_CFLAGS="${MPI_CFLAGS} ${GASNET_CFLAGS} ${GASNET_CPPFLAGS}"
  MPI_CLIBS="${MPI_CLIBS} ${GASNET_LDFLAGS} ${GASNET_LIBS}"
  OMNI_LNK_CFLAGS="${OMNI_LNK_CFLAGS} ${GASNET_LIBS}"
  OMNI_LNK_FCFLAGS="${OMNI_LNK_FCFLAGS} ${GASNET_LIBS}"
fi

AC_SUBST(GASNET_LIBS)

# Set user defined value.
if test "${userMPI_CPP}" != ""; then
  MPI_CPP="${userMPI_CPP}"
fi

if test	"${userMPI_CC}" != ""; then
  MPI_CC="${userMPI_CC}"
fi

if test "${userMPI_CFLAGS}" != ""; then
  MPI_CFLAGS="${userMPI_CFLAGS}"
fi

if test "${userMPI_FC}" != ""; then
  MPI_FC="${userMPI_FC}"
fi

if test "${userMPI_FCFLAGS}" != ""; then
  MPI_FCFLAGS="${userMPI_FCFLAGS}"
fi

if test "${userCFLAGS}" != ""; then
  CFLAGS="${userCFLAGS}"
else
  if test "$target" = "powerpc-hitachi-aix"; then
    CFLAGS="${CFLAGS} -maix64"
  fi
  chk=`${CC} --version | head -n1`
  case $chk in
  cc* | gcc* | Apple* | *ICC*)
    CFLAGS="${CFLAGS} -std=gnu99 -Wall -Wpointer-arith"
    ;;
  *)
    chk2=`${CC} --version 2> /dev/null | head -n2 | tail -n1`
    case $chk2 in
    pgcc*)
      CFLAGS="${CFLAGS} -c99"
    ;;
    *)
     echo "No supported C compiler was found."
     exit 1 ;;
    esac ;;
  esac
fi

# --------------------------------------------------------------------
# MPI3 for MPI FLAGS 
tmpmpi3=/tmp/tmpmpi3.$$.c
cat <<EOF > $tmpmpi3
#include <mpi.h>
MPI_VERSION
MPI_SUBVERSION
EOF
MPI_VERSION=`${MPI_CPP} ${MPI_CPPFLAGS} $tmpmpi3 | tail -2 | head -1`
MPI_SUBVERSION=`${MPI_CPP} ${MPI_CPPFLAGS} $tmpmpi3 | tail -1`
if test $MPI_VERSION -ge 3; then
  OMNI_X2X_FLAGS="${OMNI_X2X_FLAGS} -fasync"
  doMPI3=yes
else
  doMPI3=no
fi
rm -f $tmpmpi3
AM_CONDITIONAL(MPI3, test "x${doMPI3}" = "xyes")

# --------------------------------------------------------------------
# set userflags
if test "${userFCFLAGS}" != ""; then
  FCFLAGS="${userFCFLAGS}"
fi

if test "${userMPI_CPPFLAGS}" != ""; then
  MPI_CPPFLAGS="${userMPI_CPPFLAGS}"
fi

if test "${userMPI_FPP}" != ""; then
  MPI_FPP="${userMPI_FPP}"
fi

if test "${userMPI_FPPFLAGS}" != ""; then
  MPI_FPPFLAGS="${userMPI_FPPFLAGS}"
fi

if test "${userMPI_CLIBS}" != ""; then
  MPI_CLIBS="${userMPI_CLIBS}"
fi

if test "${userMPI_FCLIBS}" != ""; then
  MPI_FCLIBS="${userMPI_FCLIBS}"
fi

# OpenMP
AC_SUBST(OPENMP_OPT)

# XcalableMP
AC_SUBST(XMPC_DEF)
AC_SUBST(XMPF_DEF)

# MPI C
AC_SUBST(MPI_CC)
AC_SUBST(MPI_CPP)
AC_SUBST(MPI_CPPFLAGS)
AC_SUBST(MPI_CFLAGS)
AC_SUBST(MPI_CLIBS)

# MPI F
AC_SUBST(MPI_FC)
AC_SUBST(MPI_FPP)
AC_SUBST(MPI_FPPFLAGS)
AC_SUBST(MPI_FCFLAGS)
AC_SUBST(MPI_MODINC)
AC_SUBST(MPI_FCLIBS)

# Omni C
AC_SUBST(OMNI_C2X_FLAGS)
AC_SUBST(OMNI_NTV_CFLAGS)
AC_SUBST(OMNI_LNK_CFLAGS)

# Omni F
OMNI_F_PP="$MPI_FPP $MPI_FPPFLAGS"
AC_SUBST(OMNI_F_PP)
AC_SUBST(OMNI_F2X_FLAGS)
AC_SUBST(OMNI_NTV_FCFLAGS)
AC_SUBST(OMNI_LNK_FCFLAGS)

# Omni Common
AC_SUBST(OMNI_X2X_FLAGS)

# Other options
chk=`$MPI_CC --version 2> /dev/null | head -n2 | tail -n1 | grep pgcc`
AM_CONDITIONAL(IS_PGI, test "$chk" != "")

AM_CONDITIONAL(IS_CRAY, test "$target" = "Cray-linux-gnu")
AM_CONDITIONAL(IS_SUPERUX, test "$target" = "sx-nec-superux")

AC_SUBST(IS_CRAY)
AC_SUBST(IS_SUPERUX)

# --------------------------------------------------------------------
# check ranlib & ar & nm
if test "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu"; then
    AC_CHECK_PROG([RANLIB], [ranlibpx], [ranlibpx], [:])
    AC_CHECK_PROG([AR], [arpx], [arpx], [:])
    ARFLAGS="cru"
    AC_CHECK_PROG([NM], [nmpx], [nmpx], [:])

elif test "$target" = "sx-nec-superux"; then
    AC_CHECK_PROG([RANLIB], [echo], [echo], [:])
    AC_CHECK_PROG([AR], [sxar], [sxar], [:])
    ARFLAGS="cru"
    AC_CHECK_PROG([NM], [sxnm], [sxnm], [:])

elif test "$target" = "powerpc-hitachi-aix"; then
    AC_CHECK_PROG([RANLIB], [echo], [echo], [:])
    AC_CHECK_PROG([AR], [ar], [ar], [:])
    ARFLAGS="-X32_64 cru"
    AC_CHECK_PROG([NM], [nm], [nm], [:])

elif test "$target" = "powerpc-ibm-cnk"; then
    AC_CHECK_PROG([RANLIB], [echo], [echo], [:])
    AC_CHECK_PROG([AR], [bgar], [bgar], [:])
    ARFLAGS="cru"
    AC_CHECK_PROG([NM], [nm], [nm], [:])

else
   AC_CHECK_TOOL([RANLIB], [ranlib], [:])
   AC_CHECK_TOOL([AR], [ar], [:])
   ARFLAGS="cru"
   AC_CHECK_TOOL([NM], [nm], [:])
fi

AC_SUBST(ARFLAGS)
# Check RANLIB & AR & NM
if test "${RANLIB}" = ":"; then
  AC_MSG_ERROR([${RANLIB} not found]);
fi

if test "${AR}" = ":"; then
  AC_MSG_ERROR([${AR} not found]);
fi

if test "${NM}" = ":"; then
   AC_MSG_ERROR([${NM} not found]);
fi

# --------------------------------------------------------------------
# check Perl versoin is 5
PERL5=""
AC_PATH_PROGS(PERL5, perl)

inDev=0
if test "$PERL5" = ""; then
    inDev=0
else
    inDev=1
    isPerlVersion5=no
    AC_MSG_CHECKING(Perl version is 5)
    pTmp=/tmp/.btoolPerlChk.$$
    ${PERL5} -e 'require 5;' > ${pTmp} 2>&1
    if test "$?" = "0"; then
        AC_MSG_RESULT([yes])
        isPerlVersion5=yes
    else
        AC_MSG_RESULT([no])
        isPerlVersion5=no
    fi
    rm -f ${pTmp}
    unset chk pTmp
fi
AC_SUBST(PERL5)

# --------------------------------------------------------------------
# check java

if test "$userJAVA" != ""; then
  JAVA="$userJAVA"
else
  AC_PATH_PROG(JAVA, java, java)
fi

if test	"$userJAVAC" != ""; then
  JAVAC="$userJAVAC"
else
  AC_PATH_PROG(JAVAC, javac, javac)
fi

if test "$userJAR" != ""; then
  JAR="$userJAR"
else
  AC_PATH_PROG(JAR, jar, jar)
fi

AC_SUBST(JAVA)
AC_SUBST(JAVAC)
AC_SUBST(JAR)

javadir='${prefix}/share/xcalablemp'
AC_SUBST(javadir)

# java option
# Fix: #338
#JAVA_OPT="-Xmx1024m -Xms1024m"
JAVA_OPT="-Xmx200m -Xms200m"
AC_SUBST(JAVA_OPT)

#--------------------------------------------------------------------
# endian check
AC_C_BIGENDIAN

# --------------------------------------------------------------------
# Checks for types
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_MODE_T
AC_TYPE_PID_T
AC_TYPE_UID_T
AC_STRUCT_TM

# --------------------------------------------------------------------
# Checks for header files

# check stdlib.h, stdarg.h, string.h, float.h
AC_HEADER_STDC
# check if needed sys/time.h with time.h
AC_HEADER_TIME

AC_CHECK_HEADERS([fcntl.h libintl.h malloc.h stddef.h stdint.h limits.h strings.h unistd.h errno.h locale.h sys/time.h sys/resource.h sys/wait.h sys/stat.h assert.h])

# --------------------------------------------------------------------
# Checks for library functions.

# Need this to get around rpc_malloc error in AIX
# see https://www.mail-archive.com/gsoap@yahoogroups.com/msg00385.html
if test "$target" = "powerpc-hitachi-aix"; then
  export ac_cv_func_malloc_0_nonnull=yes
  export ac_cv_func_realloc_0_nonnull=yes
fi

AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_FORK
AC_FUNC_STAT
AC_FUNC_VPRINTF
AC_FUNC_STRFTIME
AC_FUNC_ERROR_AT_LINE
AC_FUNC_STRTOD

AC_CHECK_FUNCS([bzero dup2 memset setlocale system setenv])
AC_CHECK_FUNCS([strcasecmp strchr strdup strncasecmp strrchr strstr strtol])

AC_CHECK_LIB(m, pow)
AC_CHECK_LIB(m, sqrt)

# C99 functions
AC_CHECK_FUNC(strtoul, , AC_MSG_ERROR([strtoul not found]))
AC_CHECK_FUNC(strtoll, , AC_MSG_ERROR([strtoll not found]))
AC_CHECK_FUNC(strtoull, , AC_MSG_ERROR([strtoull not found]))
AC_CHECK_FUNC(strtold, , AC_MSG_ERROR([strtold not found]))

#--------------------------------------------------------------------
#    The code below deals with several issues related to gettimeofday:
#    1. Some systems don't provide a gettimeofday function at all
#       (set NO_GETTOD if this is the case).
#    2. SGI systems don't use the BSD form of the gettimeofday function,
#       but they have a BSDgettimeofday function that can be used instead.
#    3. See if gettimeofday is declared in the <sys/time.h> header file.
#       if not, set the GETTOD_NOT_DECLARED flag so that tclPort.h can
#       declare it.
#--------------------------------------------------------------------
AC_CHECK_FUNCS(gettimeofday, , [AC_DEFINE(NO_GETTOD)])
AH_TEMPLATE(NO_GETTOD, [no gettimeofday])
AC_EGREP_HEADER(gettimeofday, sys/time.h, , AC_DEFINE(GETTOD_NOT_DECLARED))
AH_TEMPLATE(GETTOD_NOT_DECLARED, [not declared gettimeofday in sys/time.h])

# --------------------------------------------------------------------
# check type size/align

AC_CHECK_SIZEOF(unsigned char)
AC_CHECK_SIZEOF(unsigned short)
AC_CHECK_SIZEOF(unsigned int)
AC_CHECK_SIZEOF(unsigned long)
AC_CHECK_SIZEOF(unsigned long long)
AC_CHECK_SIZEOF(float)
AC_CHECK_SIZEOF(double)
AC_CHECK_SIZEOF(long double)
AC_CHECK_SIZEOF(void *)
AC_CHECK_SIZEOF(_Bool)

SIZEOF_DOUBLE=$ac_cv_sizeof_double
AC_SUBST(SIZEOF_DOUBLE)
SIZEOF_FLOAT=$ac_cv_sizeof_float
AC_SUBST(SIZEOF_FLOAT)
SIZEOF_LONG_DOUBLE=$ac_cv_sizeof_long_double
AC_SUBST(SIZEOF_LONG_DOUBLE)
SIZEOF_UNSIGNED_CHAR=$ac_cv_sizeof_unsigned_char
AC_SUBST(SIZEOF_UNSIGNED_CHAR)
SIZEOF_UNSIGNED_INT=$ac_cv_sizeof_unsigned_int
AC_SUBST(SIZEOF_UNSIGNED_INT)
SIZEOF_UNSIGNED_LONG=$ac_cv_sizeof_unsigned_long
AC_SUBST(SIZEOF_UNSIGNED_LONG)
SIZEOF_UNSIGNED_LONG_LONG=$ac_cv_sizeof_unsigned_long_long
AC_SUBST(SIZEOF_UNSIGNED_LONG_LONG)
SIZEOF_UNSIGNED_SHORT=$ac_cv_sizeof_unsigned_short
AC_SUBST(SIZEOF_UNSIGNED_SHORT)
SIZEOF_VOID_P=$ac_cv_sizeof_void_p
AC_SUBST(SIZEOF_VOID_P)
SIZEOF__BOOL=$ac_cv_sizeof__Bool
AC_SUBST(SIZEOF__BOOL)

AC_CHECK_ALIGNOF(unsigned char)
AC_CHECK_ALIGNOF(unsigned short)
AC_CHECK_ALIGNOF(unsigned int)
AC_CHECK_ALIGNOF(unsigned long)
AC_CHECK_ALIGNOF(unsigned long long)
AC_CHECK_ALIGNOF(float)
AC_CHECK_ALIGNOF(double)
AC_CHECK_ALIGNOF(long double)
AC_CHECK_ALIGNOF(void *)
AC_CHECK_ALIGNOF(_Bool)

# --------------------------------------------------------------------
# 32/64 bit int type

rm -rf ./szchk ./szchk.*
AC_MSG_RESULT([Creating size check program])
${CC} ${CPPFLAGS} ${CFLAGS} ${hasLLFlags} ./buildutils/szchk.c -o ./szchk
if test ! -x ./szchk; then
    AC_MSG_RESULT([can't create size check program. exit.])
    exit 1
fi

F2C_INT64_FLAGS=""
for i in 16 32 64
do
    AC_MSG_CHECKING([$i bit integer])
    typName=`./szchk $i`
    if test "x${typName}" = "xunknown"; then
    case $i in
    16)
        TYPE_INT16="short"
        AC_SUBST(TYPE_INT16)
        asump=${TYPE_INT16};;
    32)
        TYPE_INT32="int"
        AC_SUBST(TYPE_INT32)
        asump=${TYPE_INT32};;
    64)
        TYPE_INT64="long long int"
        AC_SUBST(TYPE_INT64)
        asump=${TYPE_INT64};;
    esac
    AC_MSG_RESULT([not supported. use $asump])
    else
    AC_MSG_RESULT([$typName])
    case $i in
    16)
        AC_DEFINE(HAS_INT16)
        AH_TEMPLATE(HAS_INT16, [has int16])
        TYPE_INT16=${typName}
        AC_SUBST(TYPE_INT16);;
    32)
        AC_DEFINE(HAS_INT32)
        AH_TEMPLATE(HAS_INT32, [has int32])
        TYPE_INT32=${typName}
        AC_SUBST(TYPE_INT32);;
    64)
        AC_DEFINE(HAS_INT64)
        AH_TEMPLATE(HAS_INT64, [has int64])
        F2C_INT64_FLAGS="-DAllow_TYQUAD"
        TYPE_INT64=${typName}
        AC_SUBST(TYPE_INT64);;
    esac
    fi
done
AC_SUBST(F2C_INT64_FLAGS)
rm -rf ./szchk ./szchk.*

# --------------------------------------------------------------------
# check size of integer enough to hold size of void *

AC_MSG_CHECKING([integer type enough to hold void pointer])
${CC} -I. ${CPPFLAGS} ${CFLAGS} ${hasLLFlags} ./buildutils/chkvoidp.c -o chkvoidp
voidPint="unknown"
if test -x ./chkvoidp; then
    voidPint=`./chkvoidp`
else
    AC_MSG_RESULT([can't create check program. exit.])
    exit 1
fi
if test "x${voidPint}" = "xunknown"; then
    AC_MSG_RESULT([can't determine????])
    exit 1
else
    AC_MSG_RESULT([$voidPint])
fi
INT_ENUFF_FOR_VOIDP=$voidPint
AC_SUBST(INT_ENUFF_FOR_VOIDP)
rm -f ./chkvoidp.* ./chkvoidp

# --------------------------------------------------------------------
# check 64 bit virtual address

AC_MSG_CHECKING([64 bit virtual address])
${CC} -I. ${CPPFLAGS} ${CFLAGS} ./buildutils/addr64.c -o ./addr64
addrIs64="unknown"
if test -x ./addr64; then
    addrIs64=`./addr64`
else
    AC_MSG_RESULT([can't create check program. exit.])
    exit 1
fi
if test "x$addrIs64" = "xunknown"; then
    AC_MSG_RESULT([can't determine????])
    exit 1
else
    if test "x$addrIs64" = "x8"; then
        AC_MSG_RESULT([yes])
        AC_DEFINE(ADDR_IS_64)
        AH_TEMPLATE(ADDR_IS_64, [])
    else
        AC_MSG_RESULT([no])
    fi
fi
rm -f ./addr64.* ./addr64

# --------------------------------------------------------------------
# has printf "%qd"

rm -f ./hasqd.* ./hasqd
AC_MSG_CHECKING([has quad-precision print format])
${CC} -I. ${CPPFLAGS} ${CFLAGS} ./buildutils/hasqd.c -o ./hasqd > /dev/null 2>&1
hasQD=no
if test -x ./hasqd; then
    ret=`./hasqd`
    if test "x$ret" = "xqd" -o "x$ret" = "x%qd"; then
        hasQD=no
    else
        hasQD=yes
    fi
fi
AC_MSG_RESULT([${hasQD}])
if test "x${hasQD}" = "xyes"; then
    AC_DEFINE(HAVE_QUAD_PRINT)
    AH_TEMPLATE(HAVE_QUAD_PRINT, [has printf "%qd"])
fi
rm -f ./hasqd.* ./hasqd

# --------------------------------------------------------------------
# quad real support

gmpInclude=/usr/include
gmpLib=/usr/lib
gmpIncOpt=""
gmpDirOpt=""
qRealPrecBits=128

AC_ARG_ENABLE(mreal,
    [  --enable-mreal                 enable multiple precision real (and real*16) support of Fortran 77 (false default)],
    [doQReal=yes], [doQReal=no])

AC_ARG_WITH(gmp-lib,
    [  --with-gmp-lib=DIR          where the GNU-MP library is for quad-real support],
    [gmpLib=$withval], [true])

AC_ARG_WITH(gmp-include,
    [  --with-gmp-include=DIR      where the GNU-MP include headers are for quad-real support],
    [gmpInclude=$withval], [true])

if test "x${doQReal}" = "xyes"; then
    if test -f ${gmpInclude}/gmp.h; then
        true
    else
        AC_MSG_RESULT([can't find ${gmpInclude}/gmp.h. exit.])
        exit 1
    fi
    gmpLibs=`ls ${gmpInclude}/libgmp.* 2> /dev/null`
    if test "x${gmpLibs}" = "x"; then
        AC_MSG_RESULT([can't find ${gmpLib}/libgmp.*. exit.])
        exit 1
    fi
    gmpIncOpt="-I${gmpInclude}"
    gmpLibOpt="-L${gmpLib} -lgmp"
fi

AC_ARG_WITH(precision,
    [  --with-precision=NUM        the default precision of real*16 in bits (128)],
    [qRealPrecBits=$withval], [qRealPrecBits=128])

# libgmp check
omniLimbT="int"
omniExpT="int"
omniQRealPrecInLimb="1"
omniQRealFillGap=""
if test "x${doQReal}" = "xyes"; then
    omniQPrec=$qRealPrecBits
    AC_DEFINE(ENABLE_QREAL)
    AH_TEMPLATE(ENABLE_QREAL, [quad real])
    AC_DEFINE_UNQUOTED(OMNI_QREAL_PREC, $omniQPrec)
    AH_TEMPLATE(OMNI_QREAL_PREC, [quad real])

    qcCC () {
    rm -f $2 ${2}.*
    ${CC} -I. ${CPPFLAGS} ${CFLAGS} -DENABLE_QREAL ${hasLLFlags} ${gmpIncOpt} $3 $1 -o $2 ${gmpLibOpt}
    if test -x $2; then
        ret=`$2 $4`
        if test "x${ret}" = "x"; then
            AC_MSG_RESULT([can't determine??. exit.])
            exit 1
        else
            rm -f $2 ${2}.*
            echo ${ret}
        fi
    else
        AC_MSG_RESULT([can't create check program. exit.])
        exit 1
    fi
    }

    # check size of mp_exp_t
    AC_MSG_CHECKING([size of GNU-MP mp_exp_t])
    eSz=`qcCC buildutils/csMPexp.c ./csMPexp`
    AC_MSG_RESULT([${eSz}])
    omniExpT=""
    case ${eSz} in
    2)
        omniExpT=${TYPE_INT16};;
    4)
        omniExpT=${TYPE_INT32};;
    8)
        omniExpT=${TYPE_INT64};;
    *)
        AC_MSG_RESULT([can't determine basic type of mp_exp_t ??. exit.])
        exit 1;;
    esac
    AC_DEFINE_UNQUOTED(OMNI_SIZEOF_QEXP_T, $eSz)
    AH_TEMPLATE(OMNI_SIZEOF_QEXP_T, [size of mp_exp_t])

    # check size of mp_limb_t
    AC_MSG_CHECKING([size of GNU-MP mp_limb_t])
    lSz=`qcCC buildutils/csMPlimb.c ./csMPlimb`
    AC_MSG_RESULT([${lSz}])
    omniLimbT=""
    case ${lSz} in
    2)
        omniLimbT=${TYPE_INT16};;
    4)
        omniLimbT=${TYPE_INT32};;
    8)
        omniLimbT=${TYPE_INT64};;
    *)
        AC_MSG_RESULT([can't determine basic type of mp_limb_t ??. exit.])
        exit 1;;
    esac
    AC_DEFINE_UNQUOTED(OMNI_SIZEOF_QLIM_T, $lSz)
    AH_TEMPLATE(OMNI_SIZEOF_QLIM_T, [size of mp_limb_t])

    # check how many mp_limb_t the GNU-MP needed for N bit precision.
    AC_MSG_CHECKING([how many mp_limb_t needed for $omniQPrec bit precision])
    omniQRealPrecInLimb=`qcCC buildutils/cMPlb.c ./cMPlb '' $omniQPrec`
    AC_MSG_RESULT([$omniQRealPrecInLimb])
    AC_DEFINE_UNQUOTED(OMNI_QLIM_LEN, $omniQRealPrecInLimb)
    AH_TEMPLATE(OMNI_QLIM_LEN, [how many mp_limb_t the GNU-MP needed for N bit precision])

    # check _omQReal_t needs a gap between _mp_exp and _mp_d[omniQRealPrecInLimb].
    AC_MSG_CHECKING([_omQReal_t needs a gap between _mp_exp and _mp_d])
    gapInByte=`expr $lSz - $eSz`
    subStr=""
    if test ${gapInByte} -gt 0; then
        case ${gapInByte} in
        2)
            # lim == int, exp == short
            subStr="${TYPE_INT16} _gap_;";;
        4)
            # lim == long long, exp == int
            subStr="${TYPE_INT32} _gap_;";;
        *)
            # lim == long long, exp == short
            changequote(<<, >>)
            subStr="char _gap_[${gapInByte}];"
            changequote([, ])
            ;;
        esac
    fi
    if test "x${subStr}" != "x"; then
        #TODO
        AC_MSG_RESULT([yes, put a "${subStr}" member in _omQReal_t.])
        omniQRealFillGap=${subStr}
        AC_DEFINE(OMNI_QREAL_NEED_GAP_MEMBER)
        AH_TEMPLATE(OMNI_QREAL_NEED_GAP_MEMBER, [])
        AC_DEFINE_UNQUOTED(OMNI_QREAL_GAP_SIZE, ${gapInByte})
        AH_TEMPLATE(OMNI_QREAL_GAP_SIZE, [GNU-MP gapInByte])
    else
        omniQRealFillGap="/* if ((sizeof(mp_exp_t) < sizeof(mp_limb_t)), here must be a gap filling member. */"
        AC_MSG_RESULT([no])
    fi

    # check libgmp has an mpf_floor(), appeared in gmp3 later.
    oLIBS=${LIBS}
    LIBS="${LIBS} ${gmpLibOpt}"
    oCFLAGS=${CFLAGS}
    CFLAGS="${CFLAGS} ${CPPFLAGS} -DENABLE_QREAL ${gmpIncOpt}"
    AC_MSG_CHECKING([mpf_floor() in $LIBS])
    hasFloor=no
    AC_TRY_LINK(
    [
        #include "gmp.h"
    ],
    [
        mpf_t a;
        mpf_t b;
        (void)mpf_floor(a, b);
    ],
    [AC_MSG_RESULT(yes); hasFloor=yes],
    [AC_MSG_RESULT(no)]
    )
    LIBS=${oLIBS}
    CFLAGS=${oCFLAGS}
    if test "x${hasFloor}" = "xno"; then
        AC_MSG_RESULT([You have to use gmp3 or later. exit.])
        exit 1
    fi
fi

AC_SUBST(doQReal)
AC_SUBST(gmpIncOpt)
AC_SUBST(gmpLibOpt)
AC_SUBST(omniQPrec)
AC_SUBST(omniExpT)
AC_SUBST(omniLimbT)
AC_SUBST(omniQRealPrecInLimb)
AC_SUBST(omniQRealFillGap)

# --------------------------------------------------------------------
# Export Omni Config Variable

if test "${prefix}" = "NONE" -o "${prefix}" = ""; then
  OMNI_HOME=/usr/local
else
  OMNI_HOME=${prefix}
fi
AC_SUBST(OMNI_HOME)
AH_TEMPLATE(OMNI_HOME, [omni compiler install directory])
AC_DEFINE_UNQUOTED(OMNI_HOME, "${OMNI_HOME}")

# If "--sysconfdir=" is set,
if test "${sysconfdir}" != '${prefix}/etc'; then
  OM_DRIVER_CONF_DIR="${sysconfdir}"
else
  OM_DRIVER_CONF_DIR="${OMNI_HOME}/etc"
fi

AC_SUBST(OM_DRIVER_CONF_DIR)
AH_TEMPLATE(OM_DRIVER_CONF_DIR, [omni compiler driver configuration file directory])
AC_DEFINE_UNQUOTED(OM_DRIVER_CONF_DIR, "${OM_DRIVER_CONF_DIR}")

OM_EXEC_C_PP="${MPI_CPP}"
AC_SUBST(OM_EXEC_C_PP)

OM_EXEC_C_COMPILER="${MPI_CC}"
AC_SUBST(OM_EXEC_C_COMPILER)

OM_EXEC_C_LINKER="${MPI_CC}"
AC_SUBST(OM_EXEC_C_LINKER)

OM_EXEC_CXX_LINKER="${CXX}"
AC_SUBST(OM_EXEC_CXX_LINKER)

OM_EXEC_F_PP="${OMNI_F_PP}"
AC_SUBST(OM_EXEC_F_PP)

OM_EXEC_F_COMPILER="${MPI_FC}"
AC_SUBST(OM_EXEC_F_COMPILER)

OM_EXEC_F_LINKER="${MPI_FC}"
AC_SUBST(OM_EXEC_F_LINKER)

INCLUDECOM=include
AC_SUBST(INCLUDECOM)

XMPPKG_CONFFILE="./release/package_param.conf"

if test -f "${XMPPKG_CONFFILE}"; then
  . "${XMPPKG_CONFFILE}"
else
  AC_MSG_WARN([a configuration file for packaging is not found.])
fi

XMPPKG_CHANGELOG_DATE=`date '+%a, %d %b %Y %H:%M:%S %z'`

AC_SUBST(XMPPKG_NAME)
AC_SUBST(XMPPKG_VERSION)
AC_SUBST(XMPPKG_SUMMARY)
AC_SUBST(XMPPKG_LICENSE)
AC_SUBST(XMPPKG_RELEASE)
AC_SUBST(XMPPKG_MAINTAINER)
AC_SUBST(XMPPKG_PREFIX)
AC_SUBST(XMPPKG_DESCRIPTION)
AC_SUBST(XMPPKG_URL)
AC_SUBST(XMPPKG_GROUP)
AC_SUBST(XMPPKG_PACKAGER)
AC_SUBST(XMPPKG_VENDOR)
AC_SUBST(XMPPKG_SECTION)
AC_SUBST(XMPPKG_PRIORITY)
AC_SUBST(XMPPKG_CHANGELOG_DATE)
AM_SUBST_NOTMAKE(XMPPKG_DESCRIPTION)

AC_CONFIG_FILES(
  Makefile
  release/debian/changelog
  release/debian/control
  release/debian/rules
  release/xcalablemp.spec
  release/make_rpm.sh
  release/make_deb.sh
)

# --------------------------------------------------------------------
ATOOL_DIR=./Driver/atool
AC_SUBST(ATOOL_DIR)

RM="rm -f"
AC_SUBST(RM)

# headers/macros
AC_CHECK_HEADER(sys/param.h, [AC_DEFINE([HAVE_SYS_PARAM_H], [1], [ ])])
AC_CHECK_HEADER(ctype.h, [AC_DEFINE([HAVE_CTYPE_H], [1], [ ])])
AC_CHECK_HEADER(stdio.h, [AC_DEFINE([HAVE_STDIO_H], [1], [ ])])
AC_CHECK_HEADER(stdarg.h, [AC_DEFINE([HAVE_STDARG_H], [1], [ ])])
AC_CHECK_HEADER(dirent.h, [AC_DEFINE([HAVE_DIRENT_H], [1], [ ])])
AC_CHECK_HEADER(time.h, [AC_DEFINE([HAVE_TIME_H], [1], [ ])])
AC_CHECK_HEADER(sys/termios.h, [AC_DEFINE([HAVE_SYS_TERMIOS_H], [1], [ ])])
AC_CHECK_HEADER(signal.h, [AC_DEFINE([HAVE_SIGNAL_H], [1], [ ])])
AC_CHECK_HEADER(netdb.h, [AC_DEFINE([HAVE_NETDB_H], [1], [ ])])
AC_CHECK_HEADER(netinet/tcp.h, [AC_DEFINE([HAVE_NETINET_TCP_H], [1], [ ])])
AC_CHECK_HEADER(sys/socket.h, [AC_DEFINE([HAVE_SYS_SOCKET_H], [1], [ ])])
AC_CHECK_HEADER(arpa/inet.h, [AC_DEFINE([HAVE_ARPA_INET_H], [1], [ ])])
AC_CHECK_HEADER(sys/ioctl.h, [AC_DEFINE([HAVE_SYS_IOCTL_H], [1], [ ])])
AC_CHECK_HEADER(pthread.h, [AC_DEFINE([HAVE_PTHREAD_H], [1], [ ])])
AC_CHECK_HEADER(stdbool.h, [AC_DEFINE([HAVE_STDBOOL_H], [1], [ ])])
AC_CHECK_HEADER(sys/resource.h, [AC_DEFINE([HAVE_SYS_RESOURCE_H], [1], [ ])])
AC_CHECK_HEADER(regex.h, [AC_DEFINE([HAVE_REGEX_H], [1], [ ])])
AC_CHECK_FUNC(strtold, [AC_DEFINE([HAVE_STRTOLD], [1], [ ])])

# Assume 32 bit for cross compilation.
AC_CHECK_SIZEOF(short int, 2)
AC_CHECK_SIZEOF(int, 4)
AC_CHECK_SIZEOF(long int, 4)
AC_CHECK_SIZEOF(long long, 8)
AC_CHECK_SIZEOF(int64_t, 8)
AC_CHECK_SIZEOF(size_t, 4)
AC_CHECK_SIZEOF(void *, 4)

oCPPFLAGS=${CPPFLAGS}
oCFLAGS=${CFLAGS}
CPPFLAGS=''
CFLAGS="${CFLAGS} -Wall -Werror"
AC_MSG_CHECKING([print format for size_t])
AC_RUN_IFELSE(
	[AC_LANG_PROGRAM(
		[[#include <stdio.h>]
		 [#include <stddef.h>]
		 [#include <string.h>]],
		[[char buf[24];]
		 [snprintf(buf, sizeof(buf), "%zu", (size_t)1);]
		 [return (strcmp("1", buf) == 0 && strlen(buf) == 1) ? ]
		 [0 : 1;]])],
	[AC_DEFINE([HAVE_PRINT_FORMAT_FOR_SIZE_T], [], [ ])
	 AC_MSG_RESULT([available.])],
	[AC_MSG_RESULT([not available.])],
	[AC_MSG_RESULT([not checkable since cross-compilation.])])
CFLAGS=${oCFLAGS}
CPPFLAGS=${oCPPFLAGS}

# where the omni is.
tmpOmniDir=''
omniDir="${OMNI_HOME}"
omniBinDir=''
omniLibDir=''
omniEtcDir="${OM_DRIVER_CONF_DIR}/"
omniIncDir=''

AC_MSG_CHECKING([the Omni is installed at "${tmpOmniDir}"])
if test -d "${tmpOmniDir}"; then
    if test -x "${tmpOmniDir}/bin/F_Front"; then
	omniDir="${tmpOmniDir}"
	AC_MSG_RESULT([yes.])
    else
	omniDir=''
	AC_MSG_RESULT([no. Assume every user sets path correctly.])
    fi
fi
if test ! -z "${omniDir}"; then
    # Don't delete the last slash.
    omniBinDir="${omniDir}"/bin/
    omniLibDir="${omniDir}"/lib/
    omniIncDir="${omniDir}"/include
fi

AC_SUBST(omniDir)
AC_SUBST(omniBinDir)
AC_SUBST(omniLibDir)
AC_SUBST(omniEtcDir)
AC_SUBST(omniIncDir)

# cpp
cppCmd=${CPP}
if test -z "${CPP}"; then
   AC_MSG_FAILURE([Can't find any usable C preprosessor.])
   exit 1
fi
AC_SUBST(cppCmd)

HOSTARCH=`./buildutils/config.guess | grep -v Linking`
CPU=`echo $HOSTARCH | awk -F- '{ print $1 }' | sed 's/\./_/g'`
OS=`echo $HOSTARCH | awk -F- '{ print $3 }' | sed 's/\./_/g'`

case $CPU in
    i*86)
        CPU=i386;;
esac

OS_CPPFLAGS='-D_REENTRANT'
OS_LDFLAGS=''
OS_LIBS=''
OS_CFLAGS=''
if test -z "${MAKE}"; then
  MAKE=make
fi

API_FLAVOR=NATA_API_POSIX
TARGET_ARCH=POSIX
case $OS in
    sunos*)
        OS=sunos;;
    solaris*)
        OS=solaris;;
    aix*)
	MAKE=gmake;; 
    freebsd*)
	OS=freebsd;;
    linux*)
	OS_CPPFLAGS="${OS_CPPFLAGS} -D_GNU_SOURCE"
	OS_LIBS="-lrt -lpthread"
	OS=linux;;
    netbsd*)
        MAKE=gmake
        OS_CPPFLAGS="${OS_CPPFLAGS} -I/usr/pkg/include"
    	OS_LDFLAGS="-L/usr/pkg/lib"
	OS_LIBS="-lpthread"
	OS=netbsd;;
    cygwin*)
	OS=cygwin;;
    mingw32ce)
	OS=windowsce
	TARGET_ARCH=WINDOWS
	API_FLAVOR=NATA_API_WIN32API;;
    mingw32*)
	case $CPU in
	    x86_64*)
		OS=windows64;;
	    i*86)
		OS=windows32;;
	    *)
		OS=windows;;
	esac
	OS_LIBS="-lpthread -loleaut32 -lws2_32 -lwinmm"
	TARGET_ARCH=WINDOWS
	API_FLAVOR=NATA_API_WIN32API;;
esac

if test "$target" = "Kcomputer-linux-gnu" -o "$target" = "FX10-linux-gnu"; then
  BLASLIB="-SSL2BLAMP"
  AC_SUBST(BLASLIB)
fi

OS_CXXFLAGS=${OS_CFLAGS}
AC_SUBST(MAKE)
AC_SUBST(OS_CPPFLAGS)
AC_SUBST(OS_CFLAGS)
AC_SUBST(OS_CXXFLAGS)
AC_SUBST(OS_LDFLAGS)
AC_SUBST(OS_LIBS)

changequote(<<, >>)
CPUDEF=NATA_CPU_`echo $CPU | tr '[a-z]' '[A-Z]'`
OSDEF=NATA_OS_`echo $OS | tr '[a-z]' '[A-Z]'`
changequote([, ])
AC_SUBST(CPUDEF)
AC_SUBST(OSDEF)
AC_SUBST(API_FLAVOR)
AC_SUBST(TARGET_ARCH)

MKRULESDIR=${TOPDIR}/Driver/atool/mk
AC_SUBST(MKRULESDIR)

AC_SUBST(CPPFLAGS)
AC_SUBST(CFLAGS)
AC_SUBST(CXXFLAGS)
AC_SUBST(LDFLAGS)

AC_CONFIG_FILES(
  Driver/atool/mk/vars.mk
  Driver/atool/Makefile
  Driver/atool/src/lib/Makefile
  Driver/atool/src/include/nata/Makefile
  Driver/atool/src/include/nata/nata_platform.h
  Driver/atool/src/cmd/Makefile
  Driver/atool/src/cmd/paths.h
)

AC_CONFIG_COMMANDS([modify-config],
	[(cd Driver/atool && ${BASH} ./mk/finalizeconfig ${OS_CPPFLAGS})],
	[OS_CPPFLAGS="${OS_CPPFLAGS}"])

AC_CONFIG_COMMANDS([includes],
	[${BASH} ./Driver/atool/mk/initheaders])

AC_CONFIG_COMMANDS([depend],
	[(${BASH} ./Driver/atool/mk/initdepend && cd ./Driver/atool && ${MAKE} depend)],
	[MAKE=${MAKE}])

#AC_CONFIG_COMMANDS([config-clean],
#	[rm -rf ./config.log ./config.status.* autom4te.cache])

unset program_transform_name

AC_OUTPUT
${MAKE} clean

# --------------------------------------------------------------------
# Set version file
VERSIONTXT="./Driver/etc/version"
if test -d .git; then
  which git 1>/dev/null 2>/dev/null
  if test $? -eq 0; then
    VERSION=`git log --date=short --pretty=format:"Last Commit Day:%ad, Hash Value:%h" -1`
    echo -n Version:${XMPPKG_VERSION}, ${VERSION} > ${VERSIONTXT}
  else
    echo -n "This omni compiler was compiled on no-git environment however git clone it." > ${VERSIONTXT}
  fi
fi

# --------------------------------------------------------------------
# Output

cat << EOF
==================================================
  -- Configuration Summary --

  PREFIX : ${prefix}

  Local Compilers:
    C compiler (CC)                      : `which $CC`
    C compiler flags (CFLAGS)            : $CFLAGS
    Fortran compiler (FC)                : `which $FC`
    Fortran compiler flags (FCFLAGS)     : $FCFLAGS
    Java application launcher (JAVA)     : `which "$JAVA"`
    Java compiler (JAVAC)                : `which $JAVAC`
    Java Archive Tool (JAR)              : `which $JAR`

  MPI Compilers:
    C preprocessor flags (MPI_CPP)              : `which $MPI_CPP`
    C preprocessor flags (MPI_CPPFLAGS)         : $MPI_CPPFLAGS
    C compiler (MPI_CC)                         : `which $MPI_CC`
    C compiler flags (MPI_CFLAGS)               : $MPI_CFLAGS
    C compiler linker flags (MPI_CLIBS)         : $MPI_CLIBS
    Fortran preprocessor (MPI_FPP)              : `which $MPI_FPP`
    Fortran preprocessor flags (MPI_FPPFLAGS)   : $MPI_FPPFLAGS
    Fortran compiler (MPI_FC)                   : `which $MPI_FC`
    Fortran compiler flags (MPI_FCFLAGS)        : $MPI_FCFLAGS
    Fortran compiler linker flags (MPI_FCLIBS)  : $MPI_FCLIBS
    Supoort MPI 3 ?                             : $doMPI3

  Note: If you want to change above compilers and their options,
        please rerun ./configure MPI_CC=...

EOF

if test "$target" != "Kcomputer-linux-gnu" -a "$target" != "FX10-linux-gnu"; then
  echo "  Options:"
  echo "    GASNet                : $use_gasnet"

  if test x${use_gasnet} = xyes; then
    echo "      Installed in        : $GASNET_PREFIX"
    echo "      Conduit             : $GASNET_CONDUIT"
  fi
  echo "    OpenACC               : $doOpenAcc"
fi

if test x${use_tca} = xyes; then
  echo "    TCA                   : yes"
fi

if test x${use_libblas} = xyes; then
  echo "    BLAS LIB              : $LIBBLAS"
fi
echo "=================================================="
