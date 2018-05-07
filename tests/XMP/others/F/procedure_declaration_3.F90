#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM main
        PROCEDURE(INTEGER), POINTER :: a
        PROCEDURE(f), POINTER :: b
        PROCEDURE(), POINTER :: i
        INTERFACE
          FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
          FUNCTION g(a, b)
            INTEGER :: g
            INTEGER :: a
            INTEGER :: b
          END FUNCTION g
        END INTERFACE

        a => f
        if(a(1).eq.2) then
          PRINT *, 'PASS 1'
        else
          PRINT *, 'NG 1'
          CALL EXIT(1)
        end if
        a => g
        if(a(1, 2).eq.4) then
          PRINT *, 'PASS 2'
        else
          PRINT *, 'NG 2'
          CALL EXIT(1)
        end if
        b => f
        if(b(1).eq.2) then
          PRINT *, 'PASS 3'
        else
          PRINT *, 'NG 3'
          CALL EXIT(1)
        end if

        i => f
        if(i(1).eq.2) then
          PRINT *, 'PASS 4'
        else
          PRINT *, 'NG 4'
          CALL EXIT(1)
        end if
        i => g
        if(i(1, 2).eq.4) then
          PRINT *, 'PASS 5'
        else
          PRINT *, 'NG 5'
          CALL EXIT(1)
        end if

      END PROGRAM main

      FUNCTION f(a)
        INTEGER :: f
        INTEGER :: a
        f = 1 + 1
      END FUNCTION f

      FUNCTION g(a, b)
        INTEGER :: g
        INTEGER :: a
        INTEGER :: b
        g = 1 + a + b
      END FUNCTION g
#else
PRINT *, 'SKIPPED'
END
#endif
