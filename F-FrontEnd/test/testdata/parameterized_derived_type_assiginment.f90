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

        TYPE, EXTENDS(t) :: ttt(l, k2)
          INTEGER, LEN :: l
          INTEGER, KIND :: k2
          CHARACTER(LEN=l) :: u
          TYPE(t(k=k2)) :: m
        END TYPE ttt
        
        TYPE(t(k=4)) :: a1
        TYPE(t(k=4)) :: b1

        TYPE(t(k=4)) :: a2
        TYPE(t(k=8)) :: b2

        TYPE(t(k=8)),POINTER :: a3
        TYPE(t(k=8)),TARGET :: b3

        TYPE(ttt(l=:, k=8, k2=8)),POINTER :: a4
        TYPE(ttt(l=8, k=8, k2=8)),TARGET :: b4

        CLASS(t(k=8)),POINTER :: a5
        TYPE(ttt(l=8, k=8, k2=8)),TARGET :: b5

        a1 = b1
        !a1 = b2

        a3 => b3

        a4 => b4

        a5 => b5
        
      END PROGRAM main
