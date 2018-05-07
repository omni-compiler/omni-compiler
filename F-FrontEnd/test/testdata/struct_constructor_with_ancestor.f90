      TYPE t1
        INTEGER :: i
      END TYPE t1
      TYPE, EXTENDS(t1) :: t2
      END TYPE t2
      TYPE, EXTENDS(t2) :: t3
      END TYPE t3
      TYPE(t3) :: v = t3(t2(t1(1)))
      END
