#!/bin/sh

# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $

echo touch Makefile.in
find . -name Makefile.in -exec touch \{\} \;
sleep 1
echo touch configure
touch ./configure
sleep 1
echo touch config.status
touch ./config.status
sleep 1
echo touch Makefile
find . -name Makefile -exec touch \{\} \;

