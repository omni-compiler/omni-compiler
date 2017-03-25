# ビルド方法
configure のオプションに以下を追加：
```
--with-argobots=path/to/argobots
```

# 新たに実装したランタイム関数一覧
* ompc_loop_divide_conquer
* ompc_do_task/ompc_do_task_if
* ompc_taskwait
* ompc_taskyield

-------------------------------------------------------------------------

README
-----
# Manual
  See http://omni-compiler.org/manual.html

-----
# Implementation Status
 Some features in the language specification are not supported in this release.
  (You can download the specification of XMP from "http://xcalablemp.org")

 See docs/STATUS-XMP.md and docs/STATUS-CAF.md

----- 
# Quick Reference for XMP
## How to install
 $ ./configure --prefix=(INSTALL PATH)
 $ make; make install
 $ export PATH=(INSTALL PATH)/bin:$PATH

## Compile
 $ xmpcc  -O2 test.c   -o test
 $ xmpf90 -O2 test.f90 -o test

## Execute
$ mpirun -np 4 ./test

-----
# Quick	Reference for OpenACC
## How to install
 $ ./configure --prefix=(INSTALL PATH) --enable-openacc --with-cuda=(CUDA PATH) 
 $ make; make install
 $ export PATH=(INSTALL PATH)/bin:$PATH

## Compile
 $ ompcc -acc -O2 test.c -o test

## Execute
$ ./test
