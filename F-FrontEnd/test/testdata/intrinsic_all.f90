      program test_all
        logical,dimension(3,4) :: larray0
        logical,dimension(5) :: larray1

        integer i

        logical                   l0
        logical,dimension(4)   :: l1
        logical,dimension(3)   :: l2

        l0 = ALL(larray0)
        l1 = ALL(larray0,1)
        l2 = ALL(larray0,2)

        l0 = ALL(larray1)
        l0 = ALL(larray1,1)

        l1 = ALL(larray0,i)
        l2 = ALL(larray0,i)

        l0 = ALL((/.true., .true., .false./))

      end program test_all
