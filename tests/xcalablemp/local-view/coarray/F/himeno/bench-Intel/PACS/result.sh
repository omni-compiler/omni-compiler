#!/bin/bash

sizespec=$1
case "$sizespec" in 
    "M") ;;
    "L") ;;
    "XL") ;;
    "all") ;;
    *) echo "usage: $0 {M|L|XL|all}";
       exit 1;;
esac


scripts="go???i"


echo '[collection]----------------------------------------------'
efiles=""
ofiles=""
for f in $scripts; do
    for fe in $f.e*; do
        if [ -f $fe ]; then              # if $fe is a file
            efiles="$efiles $fe"
            ofiles="$ofiles ${fe/$f.e/$f.o}"
        else
            echo "file $f.exxxxxxx was not found"
        fi
    done
done

echo '[check errors] --------------------------------------------'
err=0
for f in $efiles; do
    if [ -s $f ]; then              # if $fe is not an empty file
        echo "$f --- ERROR"
        err=1
    else
        echo "$f --- OK"
    fi
done
#if [ $err -eq 1 ]; then
#    exit 1
#fi

echo '[trial execution summary] --------------------------------'
for f in $ofiles; do
    awk 'BEGIN {
       filename = "'$f'"
       gosaline = 0
    }
    /^himenoBMT/ {
       split($2, work, ".")
       name = work[1]
       size = $3
       split($5, work, "/")
       split(work[2], work, "-")
       nodes = work[1]
    }
    /^  Loop executed for/ {
       nloop = $4
    }
    /^   MFLOPS/ {
       gosa = $5
       if ( gosa == "" )
           gosaline = NR + 1
    }
    NR == gosaline {
       gosa = $1
       gosaline = 0
    }
    /^  MFLOPS/ {
       mflops = $2
       time = $4
       if ("'$sizespec'" == size || "'$sizespec'" == "all")
           printf "%s %3s %5d %17s %16s %12.2f GFLOPS %12.8f ms\n", filename, size, nodes, name, gosa, mflops/1000, 1000*time/nloop
    }
    END {
    }' $f
done

