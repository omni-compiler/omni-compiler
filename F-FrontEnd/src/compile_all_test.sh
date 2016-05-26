#!/bin/sh

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
#OMNI_HOME=${work}
#export OMNI_HOME
frontend=${work}/F-FrontEnd/src/F_Front
backend=${work}/F-BackEnd/bin/F_Back
nativecomp=gfortran
tmpdir=${work}/compile
testdata=$work/F-FrontEnd/test/testdata

chmod +x ${backend}
if test ! -e "${tmpdir}"; then
    mkdir -p "${tmpdir}"
fi

cd "${tmpdir}" || exit 1

ulimit -t 10

echo > errors.txt

for f in $testdata/*.f $testdata/*.f90; do
    b=`basename $f`
    errOut=${b}.out
    xmlOut=${b}.xml
    decompiledSrc=${b}.dec.f90
    binOut=${b}.o
    fOpts=''
    if test -f ${f}.options; then
        fOpts=`cat ${f}.options`
    fi
    ${frontend} ${F_FRONT_TEST_OPTS} ${fOpts} -I ${testdata} ${f} \
        -o ${xmlOut} > ${errOut} 2>&1
    if test $? -eq 0; then
        ${backend} ${xmlOut} -o ${decompiledSrc} >> ${errOut} 2>&1
        if test $? -eq 0; then
            ${nativecomp} -c ${decompiledSrc} -o ${binOut} >> ${errOut} 2>&1
            if test $? -eq 0; then
                echo ok: ${b}
            else
                echo --- failed native: ${b} | tee -a errors.txt
            fi
        else
            echo --- failed backend: ${b} | tee -a errors.txt
        fi
    else
        echo --- failed frontend: ${b} | tee -a errors.txt
    fi
done

