#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM main
        TYPE t
          INTEGER :: v = 0
          PROCEDURE(f), POINTER, NOPASS :: u => null()
          PROCEDURE(h), POINTER, PASS :: w
        END TYPE t
        INTERFACE
          FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE
        INTERFACE
          FUNCTION h(a)
            IMPORT t
            INTEGER :: h
            CLASS(t) :: a
          END FUNCTION h
        END INTERFACE
        TYPE(t) :: v
        INTEGER ret
        v%u => g
       !v%w => h
        v%w => i
        v%v = v%u(1)
        ret = v%w()
        if(ret.eq.2) then
          PRINT *, 'PASS'
        else
          PRINT *, 'NG'
          CALL EXIT(1)
        end if
      CONTAINS
        FUNCTION g(a)
          INTEGER :: g
          INTEGER :: a
          g = a + 1
        END FUNCTION g
        FUNCTION i(a)
          INTEGER :: i
          CLASS(t) :: a
          i = a%v
        END FUNCTION i
      END PROGRAM main
#else
PRINT *, 'SKIPPED'
END
#endif
