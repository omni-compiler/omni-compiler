      PROGRAM main
        REAL(KIND=4) :: integer
        REAL(KIND=4) :: logical
        REAL(KIND=4) :: real
        REAL(KIND=4) :: complex
        REAL(KIND=4) :: type
        REAL(KIND=4) :: class
        REAL(KIND=4), DIMENSION(3) :: a
        a = [1.0, 2.0, 3.0]
        a = [REAL(KIND=4) :: 1.0, 2.0, 3.0]
        a = (/1.0, 2.0, 3.0/)
        a = (/REAL(KIND=4) :: 1.0, 2.0, 3.0/)

        a = (/ integer, 1.0, 2.0/)
        a = (/ logical, 1.0, 2.0/)
        a = (/ real, 1.0, 2.0/)
        a = (/ complex, 1.0, 2.0/)
        a = (/ type, 1.0, 2.0/)
        a = (/ class, 1.0, 2.0/)
      END PROGRAM main
