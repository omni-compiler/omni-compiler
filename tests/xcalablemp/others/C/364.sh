#!/bin/sh

if [ -d __omni_tmp__ ]; then 
    echo "PASS"
    exit 0
else 
    echo "ERROR"
    exit 1
fi
