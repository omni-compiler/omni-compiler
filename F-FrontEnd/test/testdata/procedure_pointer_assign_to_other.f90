      SUBROUTINE s(p)
        PROCEDURE(t), POINTER :: p
        PROCEDURE(t), POINTER :: q
        q => p
       CONTAINS
        SUBROUTINE t
        END SUBROUTINE t
        SUBROUTINE u(p)
          PROCEDURE(t), POINTER :: p
        END SUBROUTINE u
      END SUBROUTINE s
