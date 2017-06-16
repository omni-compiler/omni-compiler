#!/bin/sh
XMP_NUM_THREADS=1  ./cholesky_task 4096 128 1
XMP_NUM_THREADS=2  ./cholesky_task 4096 128 1
XMP_NUM_THREADS=4  ./cholesky_task 4096 128 1
XMP_NUM_THREADS=8  ./cholesky_task 4096 128 1
XMP_NUM_THREADS=16  ./cholesky_task 4096 128 1
XMP_NUM_THREADS=32  ./cholesky_task 4096 128 1
XMP_NUM_THREADS=48  ./cholesky_task 4096 128 1




