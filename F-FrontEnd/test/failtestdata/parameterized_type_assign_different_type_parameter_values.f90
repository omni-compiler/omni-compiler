      PROGRAM MAIN
        TYPE t(k, l)
          INTEGER , KIND :: k
          INTEGER , LEN :: l
          INTEGER (KIND=k) :: v
          CHARACTER (LEN=l) :: c
        END TYPE t
        TYPE(t(k=4, l=4)) :: a
        TYPE(t(k=8, l=8)) :: b
        a = b
        a%v = b%v
      END PROGRAM MAIN
