      MODULE user_defined_operator
        TYPE t
           INTEGER n
        END TYPE t

        INTERFACE OPERATOR(+)
           MODULE PROCEDURE bin_ope
        END INTERFACE OPERATOR(+)

        INTERFACE ASSIGNMENT(=)
           MODULE PROCEDURE asg_ope
        END INTERFACE ASSIGNMENT(=)

      CONTAINS
        FUNCTION bin_ope(a, b)
          TYPE(t),INTENT(in)::a, b
          TYPE(t) :: bin_ope
          bin_ope = t(1)
        END FUNCTION bin_ope

        SUBROUTINE asg_ope(a, b)
          TYPE(t),INTENT(inout)::a
          TYPE(t),INTENT(in)::b
        END SUBROUTINE asg_ope
      END MODULE user_defined_operator
