#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM main
        TYPE t
          INTEGER :: v = 1
          PROCEDURE(p),PASS(a),POINTER :: p => p
        END TYPE t
        TYPE(t) :: o1, ret
        o1 = t(2)
        ret = o1%p()
        if(ret%v.eq.2) then
          PRINT *, 'PASS'
        else
          PRINT *, 'NG'
          CALL EXIT(1)
        end if
      CONTAINS
        FUNCTION p(a)
          CLASS(t) :: a
          TYPE(t) :: p
          p = a
          a%v = 3
        END FUNCTION p
      END PROGRAM main
#else
PRINT *, 'SKIPPED'
END
#endif
