! main program
      !! src: libxmpf/src/xmpf_misc.c
      !! incl. atexit, _XMP_init, FJMPI_Rdma_init, etc.
      call xmpf_init_all_

      !! Traverse subroutines xmpf_traverse_xxxx shown bellow are
      !! automatically made from the input program files and written
      !! in /tmp/omni_traverse_nnnn.f90 by omni_traverse script.
      call xmpf_traverse_module
#if defined(_XMP_COARRAY_GASNET) || defined(_XMP_COARRAY_FJRDMA)
      call xmpf_traverse_countcoarray
      call xmpf_coarray_malloc_pool
      call xmpf_traverse_initcoarray
#endif

      !! user's main program converted into a subroutine
      call xmpf_main

      !! src: libxmpf/src/xmpf_misc.c
      !! incl. FJMPI_Rdma_finalize(), _XMP_finalize, etc.
      call xmpf_finalize_all_
      end
