#!/bin/sh

verbose=0
trans=0
while [ ! -z "$1" ]; do
    case $1 in
        "-v" ) verbose=1 ;;
        "-d" ) shift; testdata=$1 ;;
        "-t" ) trans=1 ;;
        "--help" | "-?" | "-h") cat<<EOF
${0}:
	-v		run verbosely.
	-d		specify test data directory (default: ../../F-FrontEnd/test/testdata).
	-t		transform coarray statement to xmp subroutine call statement
	--help|-?	show this help.
EOF
        exit 1;;
        *) : ;;
    esac
    shift 1
done


abspath() {
    __cwd=`pwd`
    cd $1 > /dev/null 2>&1
    if test $? -eq 0; then
        /bin/pwd
    fi
    cd ${__cwd}
    unset __cwd
}


work=`abspath ../..`
OMNI_HOME=${work}
export OMNI_HOME
if test -z "${OMNI_JAVA}"; then
	OMNI_JAVA=java
fi
export OMNI_JAVA
frontend="${work}/F-FrontEnd/src/F_Front"
frontendOpt="-fintrinsic-xmodules-path ${OMNI_HOME}/F-FrontEnd/src/fincludes"
backend="${work}/F-BackEnd/bin/F_Back"
backendOpt=""
nativecomp="gfortran"
nativecompOpt="-fcoarray=single"

if test ${trans} -eq 1; then
    frontendOpt="${frontendOpt} -M${OMNI_HOME}/libxmpf/src/"
    nativecompOpt="${nativicompOpt} -I${OMNI_HOME}/libxmpf/src/"
else
    frontendOpt="${frontendOpt} -fno-xmp-coarray"
    backendOpt="${backendOpt} -fcoarray-no-use-statement"
fi

tmpdir=${work}/compile
if test -z "${testdata}"; then
    testdata=$work/F-FrontEnd/test/testdata
else
    testdata=`abspath $testdata`
fi

chmod +x ${backend}
if test ! -e "${tmpdir}"; then
    mkdir -p "${tmpdir}"
fi

cd "${tmpdir}" || exit 1

ulimit -t 10

echo > errors.txt

for f in `find -L ${testdata} -type f -a -name '*.f' -o -name '*.f90' -o -name '*.f08' | sort | xargs` ; do
    b=`basename $f`
    errOut=${b}.out
    xmlOut=${b}.xml
    decompiledSrc=${b}.dec.f90
    binOut=${b}.o
    executableOut=${b}.bin
    expectedOut=`echo ${f} | sed -e 's_/enabled/_/result/_g' -e 's_/tp/_/result/_g' -e 's/.f90$/.res/g' -e 's/.f$/.res/g' -e 's/.f08$/.res/g'`
    executeResult=${b}.res
    skipNative=${f}.skip.native
    fOpts=''
    if test -f ${f}.options; then
        fOpts=`cat ${f}.options`
    fi
    if test -f ${f}.native.options; then
        additionalNativeOpts=`cat ${f}.native.options`
    fi
    ${frontend} ${frontendOpt} ${F_FRONT_TEST_OPTS} ${fOpts} -I ${testdata} ${f} \
        -o ${xmlOut} > ${errOut} 2>&1
    if test $? -eq 0; then
        ${backend} ${backendOpt} ${xmlOut} -o ${decompiledSrc} >> ${errOut} 2>&1
        if test $? -eq 0; then
            if test ! -e "${skipNative}" ; then
                ${nativecomp} ${nativecompOpt} -c ${decompiledSrc} -o ${binOut} >> ${errOut} 2>&1
                if test $? -eq 0; then
                    if test ! -z ${expectedOut} && test -e ${expectedOut}; then
                        if test `nm ${binOut} | awk '{print $3}' | grep -c main 2>&1` -gt 0; then
                            ${nativecomp} -o ${executableOut} ${binOut} 2>> ${errOut}
                            if test $? -eq 0; then
                                ./${executableOut} > ${executeResult} 2>> ${errOut}

                                if test $? -eq 0; then
                                    diff -w -B -I '^[[:space:]]*#' -I '^[[:space:]]*//' ${executeResult} ${expectedOut} > /dev/null 2>&1

                                    if test $? -eq 0; then
                                        echo "--- ok (with_expected_output): ${b}"
                                    else
                                        echo --- failed unexpected_result: ${b} | tee -a errors.txt
                                    fi
                                else
                                    echo --- failed execution: ${b} | tee -a errors.txt
                                fi
                            else
                                echo "--- failed link: ${b}" | tee -a errors.txt
                            fi
                        else
                            echo "--- ok : ${b}"
                        fi
                    else
                        echo "--- ok : ${b}"
                    fi
                else
                    echo "--- failed native: ${b}" | tee -a errors.txt
                fi
            else
                echo "--- ok(skip_native) : ${b}"
            fi
        else
            echo "--- failed backend: ${b}" | tee -a errors.txt
        fi
    else
        echo "--- failed frontend: ${b}" | tee -a errors.txt
    fi
    if test ${verbose} -eq 1; then
        cat ${errOut}
    fi
done

