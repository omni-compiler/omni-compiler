
      PROGRAM test_sequence
        TYPE t(k)
          INTEGER , KIND :: k
          INTEGER (KIND=k) :: v
        END TYPE t
        TYPE, EXTENDS(t) :: tt
           INTEGER (KIND=k) :: u
        END type tt
      END PROGRAM test_sequence
