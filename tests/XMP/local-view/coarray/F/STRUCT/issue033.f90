  program struct_pointer

    type g1
       integer a(3)
       type(g1),pointer :: p
    end type g1

    type(g1), save ::  cs1[*]

    ierr = 0
    n_alloced = xmpf_coarray_allocated_bytes()
    n_necessary = sizeof(cs1)

    me = this_image()
    if (n_alloced == n_necessary) then
       write(*,100) me
    else if (n_alloced > n_necessary) then
       write(*,101) me, n_alloced, n_necessary
    else
       write(*,102) me, n_alloced, n_necessary
    endif

100 format("[", i0, "] OK. perfect")
101 format("[", i0, "] OK, but allocated size (", i0,       &
         " bytes) is larger than necessary size (", i0,     &
         " bytes).")
102 format("[", i0, "] NG. Allocated size (", i0,           &
         " bytes) is smaller than necessary size (", i0,    &
         " bytes).")

  end program
