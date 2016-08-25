      PROGRAM MAIN
        USE nested_types
        TYPE, EXTENDS (t1) :: t2
           INTEGER :: t
        END type t2

        TYPE(t2) :: a

        a%v = 1
        a%u = 2
        a%t = 3
      END PROGRAM MAIN
