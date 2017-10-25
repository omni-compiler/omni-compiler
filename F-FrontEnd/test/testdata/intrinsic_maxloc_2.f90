      program test_maxloc
        INTEGER,           DIMENSION(5,4,3)     :: array1
        INTEGER,           DIMENSION(5)         :: array2

        INTEGER                                 :: i0
        INTEGER,           DIMENSION(4,3)       :: i1
        INTEGER,           DIMENSION(5,3)       :: i2
        INTEGER,           DIMENSION(5,4)       :: i3

        i1 = MAXLOC(array1, 1)
        i2 = MAXLOC(array1, 2)
        i3 = MAXLOC(array1, 3)

        i  = MAXLOC(array2, 1)

      end program test_maxloc
