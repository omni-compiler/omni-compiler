      MODULE m
        TYPE, BIND(C) :: t
          INTEGER :: v
        END TYPE t
       CONTAINS
        FUNCTION f() BIND(C)
          TYPE(t) :: f
          f%v = 1
        END FUNCTION f
      END MODULE m
