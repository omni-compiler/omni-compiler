      TYPE :: t
        INTEGER :: v
        INTEGER :: u = 1
      END TYPE t
      TYPE, EXTENDS(t) :: tt
        INTEGER :: w
      END TYPE tt
      TYPE(t), PARAMETER :: a = t(1)
      TYPE(tt) :: b = tt(a, 2)
      TYPE(tt) :: c = tt(t(1), 2)
      END
