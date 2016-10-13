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

## On the K computer, FX100, and FX10
* The number of coarrays in an application is 508 or less
* An application cannot be used in more than 82,944 processes
* Post tag value is between 0 and 14 (0 <= tag <= 14)

## The in and out clauses of the gmove directives
* The target of the gmove in/out directives must be declared as a module variable or
  a variable with the SAVE attribute in XMP/F, or as an external variable in XMP/C
