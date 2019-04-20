#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      PROGRAM main
        USE m__derived_type_and_generics_in_module_2
        REAL :: r, u

        a = t(1)
        r = t(2.0)
        b = s(1)
        u = s(2.0)

        if(a%v+r.eq.3.and.b%v+u.eq.3) then
          PRINT *, 'PASS'
        else
          PRINT *, 'NG'
          CALL EXIT(1)
        end if

#else
PRINT *, 'SKIPPED'
#endif
      END PROGRAM main
