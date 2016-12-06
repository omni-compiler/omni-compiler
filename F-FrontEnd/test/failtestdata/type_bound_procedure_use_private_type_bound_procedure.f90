          MODULE m
            TYPE :: t
              INTEGER :: v
            CONTAINS
              PROCEDURE,PASS,PUBLIC :: show => show
              PROCEDURE,PASS,PRIVATE :: inc => inc
            END TYPE t
          CONTAINS
            SUBROUTINE show(a)
              CLASS(t) :: a
              PRINT *, a%v
            END SUBROUTINE show
            FUNCTION inc(a)
              CLASS(t) :: a
              a%v = a%v + 1
              inc = a%v
            END FUNCTION
          END MODULE m

          PROGRAM MAIN
            USE m
            INTEGER :: i
            CLASS(t), POINTER :: a
            TYPE(t), TARGET :: b
            b = t(v=1)
            a => b
            CALL a%show()
            i = a%inc()
            CALL a%show()
          END PROGRAM MAIN
