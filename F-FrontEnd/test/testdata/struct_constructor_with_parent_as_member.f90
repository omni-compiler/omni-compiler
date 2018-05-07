       TYPE t
         INTEGER :: i
       END TYPE t
       TYPE,EXTENDS(t) :: tt
         INTEGER :: j
         TYPE(t) :: k
       END TYPE tt
       TYPE(tt) :: v = tt(1, 2, t(1))
       END
