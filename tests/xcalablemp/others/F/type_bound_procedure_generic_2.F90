#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      MODULE m_type_bound_procedure_generic_2
        INTEGER :: v
        TYPE t
         CONTAINS
           PROCEDURE, NOPASS, PUBLIC :: f
           PROCEDURE, NOPASS, PUBLIC :: g
           GENERIC :: p => f, g
        END TYPE t
        TYPE, EXTENDS(t) :: tt
         CONTAINS
           PROCEDURE, NOPASS, PUBLIC :: g => h
        END TYPE tt
      CONTAINS
        SUBROUTINE f(i)
          INTEGER :: i
          v = v + i
!         PRINT *, "call F"
        END SUBROUTINE f
        SUBROUTINE g(r)
          REAL :: r
          if (v.eq.2) then
            print *, 'PASS 1'
          else
            print *, 'NG 1 : ', v, ', should be 2'
            call exit(1)
          endif
!         PRINT *, "call G"
        END SUBROUTINE g
        SUBROUTINE h(r)
          REAL :: r
          if (v.eq.3) then
            print *, 'PASS 2'
          else
            print *, 'NG 2 : ', v, ', should be 3'
            call exit(1)
          endif
!         PRINT *, "call H"
        END SUBROUTINE h
      END MODULE m_type_bound_procedure_generic_2

      PROGRAM main
        USE m_type_bound_procedure_generic_2
        TYPE(t), TARGET :: a
        CLASS(t), POINTER :: b
        TYPE(tt), TARGET :: c
        v = 1
        b => a
        CALL b%p(1)
        CALL b%p(1.2)
        v = 1
        b => c
        CALL b%p(2)
        CALL b%p(2.2)
      END PROGRAM main
#else
print *, 'SKIPPED'
end
#endif
