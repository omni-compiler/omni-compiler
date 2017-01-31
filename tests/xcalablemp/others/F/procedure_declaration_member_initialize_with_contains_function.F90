#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM main
        TYPE t
          INTEGER :: v = 1
          PROCEDURE(f),PASS(a),POINTER :: p => g
        END TYPE t
        INTERFACE
          FUNCTION f(a)
            IMPORT t
            CLASS(t) :: a
            TYPE(t) :: f
          END FUNCTION f
        END INTERFACE
        TYPE(t) :: o1, ret
        o1%v = 2
        ret = o1%p()
        if(ret%v.eq.2) then
          PRINT *, 'PASS'
        else
          PRINT *, 'NG'
          CALL EXIT(1)
        end if
      CONTAINS
        FUNCTION g(a)
          CLASS(t) :: a
          TYPE(t) :: g
          g = a
          a%v = 3
        END FUNCTION g
      END PROGRAM main
#else
PRINT *, 'SKIPPED'
END
#endif
