      PROGRAM MAIN
        TYPE :: t
        END TYPE t
        CLASS(t),POINTER :: a
        TYPE(t),TARGET :: b
        CLASS(t),ALLOCATABLE :: c
        a => b
      CONTAINS
        SUBROUTINE SUB(a)
          CLASS(t),POINTER :: a
        END SUBROUTINE SUB
      END PROGRAM MAIN
