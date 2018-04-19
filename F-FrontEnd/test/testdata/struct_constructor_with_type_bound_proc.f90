       MODULE m
         TYPE t
           INTEGER :: i
          CONTAINS
           PROCEDURE :: sub
         END TYPE t
         TYPE, EXTENDS(t) :: tt
           INTEGER :: j
         END TYPE tt
        CONTAINS
         SUBROUTINE sub(a)
           CLASS(t), INTENT(IN) :: a
         END SUBROUTINE sub
       END MODULE m

       PROGRAM main
         USE m
         TYPE(tt) :: v = tt(1,2)
       END PROGRAM main
