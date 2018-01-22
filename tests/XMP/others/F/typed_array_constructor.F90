      PROGRAM main
#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
        REAL(KIND=4) :: integer
        REAL(KIND=4) :: logical
        REAL(KIND=4) :: real
        REAL(KIND=4) :: complex
        REAL(KIND=4) :: type
        REAL(KIND=4) :: class = 3.0
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
        if(a(1).eq.3.0) then
          print *, 'PASS'
        else
          print *, 'NG'
          call exit(1)
        end if
#else
  PRINT *, 'SKIPPED'
#endif
      END PROGRAM main
