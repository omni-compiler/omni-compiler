      PROGRAM MAIN
        TYPE :: t
        END TYPE t
        TYPE,EXTENDS(t) :: tt
        END type tt
        CLASS(t),POINTER :: a
        TYPE(t),TARGET :: b
        TYPE(tt),TARGET :: c
        a => b
        a => c
      END PROGRAM MAIN
