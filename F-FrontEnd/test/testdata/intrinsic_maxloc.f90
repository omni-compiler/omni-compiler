      program test_maxloc
        INTEGER,           DIMENSION(5,4)     :: arrayI
        REAL,              DIMENSION(5,4)     :: arrayR
        CHARACTER(LEN=10), DIMENSION(5,4)     :: arrayC

        LOGICAL,           DIMENSION(5,4)     :: mask

        INTEGER,           DIMENSION(2)       :: i0
        REAL,              DIMENSION(2)       :: r0
        CHARACTER(LEN=10), DIMENSION(2)       :: s0

        INTEGER,           DIMENSION(4)       :: i1
        REAL,              DIMENSION(4)       :: r1
        CHARACTER(LEN=10), DIMENSION(4)       :: s1

        ! MAXLOC(ARRAY)
        i0 = MAXLOC(arrayI)
        r0 = MAXLOC(arrayR)
        s0 = MAXLOC(arrayC)

        ! MAXLOC(ARRAY,MASK)
        i0 = MAXLOC(arrayI, mask)
        r0 = MAXLOC(arrayR, mask)
        s0 = MAXLOC(arrayC, mask)

        ! MAXLOC(ARRAY,MASK,KIND)
        i0 = MAXLOC(arrayI, mask, 8)
        r0 = MAXLOC(arrayR, mask, 8)
        s0 = MAXLOC(arrayC, mask, 8)

        ! MAXLOC(ARRAY,MASK,KIND,BACK)
        i0 = MAXLOC(arrayI, mask, 8, .TRUE.)
        r0 = MAXLOC(arrayR, mask, 8, .TRUE.)
        s0 = MAXLOC(arrayC, mask, 8, .TRUE.)

        ! MAXLOC(ARRAY,DIM)
        i1 = MAXLOC(arrayI, 1)
        r1 = MAXLOC(arrayR, 1)
        s1 = MAXLOC(arrayC, 1)

        ! MAXLOC(ARRAY,DIM,MASK)
        i1 = MAXLOC(arrayI, 1, mask)
        r1 = MAXLOC(arrayR, 1, mask)
        s1 = MAXLOC(arrayC, 1, mask)

        ! MAXLOC(ARRAY,DIM,MASK,KIND)
        i1 = MAXLOC(arrayI, 1, mask, 8)
        r1 = MAXLOC(arrayR, 1, mask, 8)
        s1 = MAXLOC(arrayC, 1, mask, 8)

        ! MAXLOC(ARRAY,DIM,MASK,KIND,BACK)
        i = MAXLOC(arrayI, 0, mask, 8, .TRUE.)
        r = MAXLOC(arrayR, 0, mask, 8, .TRUE.)
        s = MAXLOC(arrayC, 0, mask, 8, .TRUE.)

      end program test_maxloc
