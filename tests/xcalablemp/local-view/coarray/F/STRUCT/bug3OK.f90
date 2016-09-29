  program get1

    type g0
       integer a(3)
    end type g0

    type g1
       integer a(3)
       type(g0),pointer :: p
    end type g1

    type(g1), save ::  cs1[*]

  end program
