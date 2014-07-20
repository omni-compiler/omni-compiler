#!/bin/sh
FC="om3-fdebug --debug --tempdir . -Topenmp -c --Wx-l"

function comp_ok {
    echo $1
    ${FC} $1 > tmp.log 2>&1
    if [ "$?" != "0" ]; then
        cat tmp.log
    fi
} 

function comp_err {
    echo $1
    ${FC} $1 > tmp.log 2>&1
    if [ "$?" = "0" ]; then
        cat tmp.log
        echo "*** $1 : expected error but ok"
    fi
} 

exts="F F90"

for e in $exts; do
    for f in `ls expect_ok/*.$e`; do
        comp_ok $f
    done
done

for e in $exts; do
    for f in `ls expect_err/*.$e`; do
        comp_err $f
    done
done

