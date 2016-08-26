      PROGRAM MAIN
        TYPE st(lg)
           INTEGER, LEN :: lg
           CHARACTER(LEN=lg), POINTER :: ch
        END TYPE st

        CHARACTER(LEN=10), TARGET :: target
        TYPE(st(lg=:)) :: a
        TYPE(st(lg=:)):: b
        b = st(lg=:)(target)

      END PROGRAM MAIN
