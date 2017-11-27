      program test_maxloc
        INTEGER,           DIMENSION(5,4,3)     :: array1
        INTEGER,           DIMENSION(4,3)       :: i1
        i1 = MAXLOC(array1, LEN("a"))
      end program test_maxloc
