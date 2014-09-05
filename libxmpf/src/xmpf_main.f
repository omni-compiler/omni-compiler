! main program
      call xmpf_init_all_
      call xmpf_module_init_   ! old-fashioned
      call xmpf_init_coarray1
      call xmpf_init_coarray2
      call xmpf_main
      call xmpf_finalize_coarray
      call xmpf_finalize_all_
      end
