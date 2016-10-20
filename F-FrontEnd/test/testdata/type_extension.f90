      PROGRAM MAIN
        TYPE :: a
          INTEGER :: v
        END TYPE a
        TYPE, EXTENDS ( a ) :: b
          INTEGER :: u
        END TYPE b
        TYPE(b) :: c


        c%u = 1
        c%a = a(0)
        c%a%v = 2
        c%v = 3
      END PROGRAM MAIN
