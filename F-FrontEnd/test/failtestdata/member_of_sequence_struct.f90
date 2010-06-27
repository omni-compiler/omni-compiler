      MODULE m0
        TYPE t
           INTEGER :: a
           INTEGER :: b
        END type t
      END MODULE m0

      PROGRAM test_sequence
        USE m0
        TYPE tt
           SEQUENCE
           TYPE(t) :: v
           INTEGER :: a
        END type tt
      END PROGRAM test_sequence
