    1          MODULE m
    2            TYPE :: t
    3              INTEGER :: v
    4            CONTAINS
    5              PROCEDURE,PASS :: show => show
    6            END TYPE t
    7          CONTAINS
    8            SUBROUTINE show(a)
    9              CLASS(t) :: a
   10              PRINT *, a%v
   11            END SUBROUTINE show
   12          END MODULE m
!  13
   14          MODULE mm
   15            USE m
   16            TYPE, EXTENDS(t) :: tt
   17              INTEGER :: u
   18            CONTAINS
   19              PROCEDURE,PASS :: show => show2
   20            END TYPE tt
   21          CONTAINS
   22            SUBROUTINE show2(a)
   23              CLASS(tt) :: a
   24              PRINT *, a%u
   25            END SUBROUTINE show2
   26          END MODULE mm
!  27
   28          PROGRAM MAIN
   29            USE mm
   30            CLASS(t), POINTER :: a
   31            TYPE(tt),TARGET :: b = tt(v=1, u=2)
   32            a => b
   33            CALL a%show()
   34          END PROGRAM MAIN
