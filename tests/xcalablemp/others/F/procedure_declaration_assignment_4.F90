#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      FUNCTION f(a)
        INTEGER :: f
        INTEGER :: a
        f = a
      END FUNCTION f

      SUBROUTINE sub()
        PRINT *, 'PASS 2'
      END SUBROUTINE sub

      PROGRAM main
        INTERFACE
          FUNCTION  f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
          SUBROUTINE sub()
          END SUBROUTINE sub
        END INTERFACE
        PROCEDURE(), POINTER :: i
        PROCEDURE(), POINTER :: j
        i => f
        j => sub
        if(i(10).eq.10) then
          PRINT *, 'PASS 1'
        else
          PRINT *, 'NG'
          CALL EXIT(1)
        end if
        CALL j
      END PROGRAM main
#else
PRINT *, 'SKIPPED'
END
#endif
