      PROGRAM MAIN
        USE nested_types
        TYPE(t1) :: a

        a%u = 1
        a%t0 = t0(0)
        a%t0%v = 2
        a%v = 3
      END PROGRAM MAIN
