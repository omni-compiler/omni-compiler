  program struct_pointer

    type g1
       integer a(3)
       type(g1),pointer :: p
    end type g1

    type(g1), save ::  cs1[*]

    write(*,*) xmpf_coarray_allocated_bytes()
    write(*,*) sizeof(cs1)

  end program
