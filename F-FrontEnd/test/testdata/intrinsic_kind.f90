      program main
        integer           :: result
        integer(KIND=4)   :: i0
        character(KIND=1) :: s0
        real(KIND=4)      :: r0
        logical(KIND=4)   :: l0
        complex(KIND=4)   :: c0
        integer*4         :: i1
        character*4       :: s1
        real*4            :: r1
        logical*4         :: l1
        complex*8         :: c1

        result = KIND(i0)
        result = KIND(s0)
        result = KIND(r0)
        result = KIND(l0)
        result = KIND(c0)
        result = KIND(i1)
        result = KIND(s1)
        result = KIND(r1)
        result = KIND(l1)
        result = KIND(1)
        result = KIND(1_2)
        result = KIND(' ')
        result = KIND(.true.)
      end program main
