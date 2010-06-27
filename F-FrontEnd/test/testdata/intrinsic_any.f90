      program test_any
        logical,dimension(3,4) :: larray0
        logical,dimension(5) :: larray1

        logical                   l0
        logical,dimension(4)   :: l1
        logical,dimension(3)   :: l2

        l0 = ANY(larray0)
        l1 = ANY(larray0,1)
        l2 = ANY(larray0,2)

        l0 = ANY(larray1)
        l0 = ANY(larray1,1)
      end program test_any
