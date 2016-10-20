      PROGRAM MAIN
        TYPE t
          INTEGER :: v
        END TYPE t
        INTEGER, POINTER, VOLATILE :: a0
        INTEGER, DIMENSION(3,3), TARGET, VOLATILE :: b0
        INTEGER, POINTER, VOLATILE :: a1
        TYPE(t), TARGET, VOLATILE :: b1
        TYPE(t), DIMENSION(3,3), TARGET, VOLATILE :: c1
        a0 => b0(1,1)
        a1 => b1%v
        a1 => c1(1,1)%v
      END PROGRAM MAIN
