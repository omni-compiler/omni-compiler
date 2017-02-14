#if defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      MODULE m_type_bound_procedure_5
        TYPE t
         CONTAINS
           PROCEDURE, PUBLIC :: f
           PROCEDURE, PUBLIC :: g
           GENERIC :: p => f, g
        END TYPE t
      CONTAINS
        FUNCTION f(this, i)
          INTEGER :: f
          CLASS(t) :: this
          INTEGER :: i
          f = this%p(1.0 * i)
        END FUNCTION f
        FUNCTION g(this, r)
          CLASS(t) :: this
          REAL :: r
          REAL :: g
          g = r * 3.0
        END FUNCTION g
      END MODULE m_type_bound_procedure_5

      PROGRAM main
        USE m_type_bound_procedure_5
        type(t) o1
        integer ret
        ret = o1%p(2)
        if(ret.eq.6) then
          print *, 'PASS'
        else
          print *, 'NG'
          call exit(1)
        end if
      END PROGRAM main
#else
print *, 'SKIPPED'
end
#endif

