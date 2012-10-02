      MODULE user_operation
        TYPE t
           INTEGER n
        END TYPE t

        INTERFACE OPERATOR(+)
           MODULE PROCEDURE bin_ope
        END INTERFACE OPERATOR(+)

        INTERFACE OPERATOR(-)
           MODULE PROCEDURE bin_ope
        END INTERFACE OPERATOR(-)

        INTERFACE OPERATOR(/)
           MODULE PROCEDURE bin_ope
        END INTERFACE OPERATOR(/)

        INTERFACE OPERATOR(*)
           MODULE PROCEDURE bin_ope
        END INTERFACE OPERATOR(*)

        INTERFACE OPERATOR(==)
           MODULE PROCEDURE cmp_ope
        END INTERFACE OPERATOR(==)

        INTERFACE OPERATOR(>=)
           MODULE PROCEDURE cmp_ope
        END INTERFACE OPERATOR(>=)

        INTERFACE OPERATOR(<=)
           MODULE PROCEDURE cmp_ope
        END INTERFACE OPERATOR(<=)

        INTERFACE OPERATOR(/=)
           MODULE PROCEDURE cmp_ope
        END INTERFACE OPERATOR(/=)

        INTERFACE OPERATOR(.HOGE.)
           MODULE PROCEDURE bin_ope
        END INTERFACE OPERATOR(.HOGE.)

        INTERFACE OPERATOR(.HUGA.)
           MODULE PROCEDURE cmp_ope
        END INTERFACE OPERATOR(.HUGA.)

        INTERFACE ASSIGNMENT(=)
           MODULE PROCEDURE asg_ope
        END INTERFACE ASSIGNMENT(=)

      CONTAINS
        FUNCTION bin_ope(a, b)
          TYPE(t),INTENT(in)::a, b
          TYPE(t) :: bin_ope
          bin_ope = t(1)
        END FUNCTION bin_ope

        LOGICAL FUNCTION cmp_ope(a, b)
          TYPE(t),INTENT(in)::a, b
          cmp_ope = .TRUE.
        END FUNCTION cmp_ope

        SUBROUTINE asg_ope(a, b)
          TYPE(t),INTENT(inout)::a
          TYPE(t),INTENT(in)::b
        END SUBROUTINE asg_ope
      END MODULE user_operation

