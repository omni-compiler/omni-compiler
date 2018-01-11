  program tt
    integer, parameter :: n1=1, n2=2
    character :: c[*]
    character(len=1) :: c1[*]
    character(len=2) :: c2[*]
    character(len=n1) :: cn1[*]
    character(len=n2) :: cn2[*]
    character(len=4) :: c4[*]

!!    c = c[1]       !! error
!!    c = c1[1]      !! error
    c = c2[1]
!!    c = cn1[1]     !! error
    c = cn2[1]
    c = c4[1]

  end program
