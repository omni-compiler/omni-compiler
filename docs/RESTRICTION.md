# XMP/C and XMP/Fortran
## Using GASNet for coarray, lock/unlock, and post/wait (local-view operations)
The Omni compiler uses GASNet for local-view operations, and also uses MPI library for global-view operations.
However, GASNet cannot use MPI communication except for mpi-conduit at the same time.
(In detail, please read a section "MPI Interoperability" in README of GASNet)

Therefore, when using both local-view and global-view operations at the same time,
XMP application may not be executed.
If you want to use both local-view and global-view operations in one application,
you need to use xmp_sync_all() or barrier directive to divide communication.

---
  /* ... local-view program ... */

  xmp_sync_all(&status);

  /* ... global-view program ... */

 #pragma xmp barrier

  /* ... local-view program ... */
---

## The K computer and FX10
* Coarray transfer data size is 16,777,212 (2^24-4) Byte and fewer.
* Post tag value is 0 and over && 14 and fewer (0 <= tag <= 14).
* The number of coarrays in one application is 508 and fewer.
* Onesided operations cannot be used in more than 82944 processes.

# Only XMP/Fortran
* In "use statement", only a module compiled with the Omni XMP Fortran Compiler can be used.
