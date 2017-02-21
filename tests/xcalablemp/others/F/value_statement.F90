      PROGRAM main
#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
        integer :: c,d
        c = 1
        d = 2
        call f(c,d)
        if(c.eq.1.and.d.eq.2) then
          print *, 'PASS'
        else
          print *, 'NG'
          call exit(1)
        end if
        CONTAINS
       SUBROUTINE f(a, b)
         INTEGER :: a
         VALUE :: a
         INTEGER, VALUE :: b
         a = 3
         b = 4
       END SUBROUTINE f
#else
  PRINT *, 'SKIPPED'
#endif
      END PROGRAM main
