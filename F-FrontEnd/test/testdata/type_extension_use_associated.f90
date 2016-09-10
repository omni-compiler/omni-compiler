      PROGRAM MAIN
        USE base_types
        TYPE, EXTENDS ( t0 ) :: t1
          INTEGER :: u
        END TYPE t1
        TYPE(t1) :: a

        a%u = 1
        a%t0 = t0(0)
        a%t0%v = 2
        a%v = 3
      END PROGRAM MAIN
