      PROGRAM test_sequence
        TYPE t(k, l)
          INTEGER , KIND :: k
          INTEGER , LEN :: l
          INTEGER (KIND=k+4) :: v
          CHARACTER (LEN=l+4) :: c
        END TYPE t
        TYPE(t(k=4, l=4)) :: a
        INTEGER :: b
        TYPE(t(k=4, l=:)), POINTER :: c
        b = a%v
      END PROGRAM test_sequence
