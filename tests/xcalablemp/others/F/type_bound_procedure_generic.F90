#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      MODULE m_type_bound_procedure_generic
        IMPLICIT NONE

        TYPE :: t
          INTEGER :: i
        CONTAINS
          PROCEDURE :: assign_t_from_int
          PROCEDURE :: equals_t_int
          GENERIC :: ASSIGNMENT(=) => assign_t_from_int
          GENERIC :: OPERATOR(==) => equals_t_int
        END TYPE t

      CONTAINS

        SUBROUTINE assign_t_from_int (me, i)
          IMPLICIT NONE
          CLASS(t), INTENT(OUT) :: me
          INTEGER, INTENT(IN) :: i
          me%i = i
        END SUBROUTINE assign_t_from_int

        LOGICAL FUNCTION equals_t_int (me, i)
          IMPLICIT NONE
          CLASS(t), INTENT(IN) :: me
          INTEGER, INTENT(IN) :: i
          equals_t_int = (me%i == i)
        END FUNCTION equals_t_int

      END MODULE m_type_bound_procedure_generic

      use m_type_bound_procedure_generic

      type(t) o1

      o1 = 10

      if (o1==10) then
        print *,'PASS'
      else
        print *,'NG : ', o1, ', should be 10'
        call exit(1)
      end if

      end
#else
print *, 'SKIPPED'
end
#endif
