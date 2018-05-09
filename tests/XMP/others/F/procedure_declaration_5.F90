#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM main
        TYPE t
          INTEGER :: v
        END TYPE t
        PROCEDURE(TYPE(t)), POINTER :: g
        TYPE(t) RET
        g => f
        RET = f()
        if(RET%v.eq.10) then
          PRINT *, 'PASS'
        else
          PRINT *, 'NG'
          CALL EXIT(1)
        end if
      CONTAINS
        FUNCTION f()
          TYPE(t) f
          TYPE(t) o
          o = t(10)
          f = o
        END FUNCTION f
      END PROGRAM main
#else
PRINT *, 'SKIPPED'
END
#endif
