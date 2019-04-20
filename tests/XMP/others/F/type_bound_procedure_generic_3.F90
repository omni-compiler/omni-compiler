#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      MODULE m_type_bound_procedure_generic_3
        INTEGER :: v
        TYPE t
         CONTAINS
           PROCEDURE, NOPASS, PUBLIC :: f
           PROCEDURE, NOPASS, PUBLIC :: g
           PROCEDURE, NOPASS, PUBLIC :: h
           GENERIC :: p => f, g, h
        END TYPE t
      CONTAINS
        SUBROUTINE f(i)
          INTEGER(KIND=4) :: i
          v = i
!         PRINT *, "call F"
        END SUBROUTINE f
        SUBROUTINE g(r)
          INTEGER(KIND=8) :: r
          v = v + r
!         PRINT *, "call G"
        END SUBROUTINE g
        SUBROUTINE h(r)
          INTEGER(KIND=16) :: r
          v = v + r
!         PRINT *, "call H"
        END SUBROUTINE h
      END MODULE m_type_bound_procedure_generic_3

      PROGRAM main
        USE m_type_bound_procedure_generic_3
        TYPE(t), TARGET :: a
        CLASS(t), POINTER :: b
        b => a
        CALL b%p(1_4)
        CALL b%p(1_8)
        CALL b%p(1_16)
        if (v.eq.3) then
          print *, 'PASS'
        else
          print *, 'NG : ', v, ', should be 3'
          call exit(1)
        endif
        !CALL b%p(1_32)
      END PROGRAM main
#else
print *, 'SKIPPED'
end
#endif
