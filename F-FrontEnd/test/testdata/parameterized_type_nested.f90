      PROGRAM main
        TYPE t(k)
          INTEGER, KIND :: k
          INTEGER(KIND=k) :: v
        END TYPE t
        TYPE, EXTENDS(t) :: tt(l, k2)
          INTEGER, LEN :: l
          INTEGER, KIND :: k2
          CHARACTER(LEN=l) :: u
          TYPE(t(k=k2)) :: m
        END TYPE tt
        
        TYPE(t(k=4)) :: u

        TYPE(tt(k=4, l=4 ,k2=8)) :: v 
        INTEGER(KIND=4) :: a1
        INTEGER(KIND=8) :: a2
        a1 = u%v
        a1 = v%t%v
        a2 = v%m%v
        
      END PROGRAM main
