#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM main
        PROCEDURE(REAL(KIND=8)), POINTER :: p

        INTERFACE
          FUNCTION f(a)
            REAL(KIND=8) :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE

        REAL(KIND=8) :: r

        p => f
        r = p(1)
        if(r.eq.1) then
          PRINT *, 'PASS'
        else
          PRINT *, 'NG'
          CALL EXIT(1)
        end if
      END PROGRAM main

      FUNCTION f(a)
        REAL(KIND=8) :: f
        INTEGER :: a
        f = a
      END FUNCTION f

#else
PRINT *, 'SKIPPED'
END
#endif
