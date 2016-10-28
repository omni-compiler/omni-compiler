      PROGRAM MAIN
        TYPE :: t
        END TYPE t
        TYPE :: u
        END TYPE u
        CLASS(*),POINTER :: a
        TYPE(t),TARGET :: b
        TYPE(u),TARGET :: c
        a => b
        a => c
      END PROGRAM MAIN
