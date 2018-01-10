      PROGRAM main
        TYPE t
         INTEGER :: i = 0
        END TYPE
        CLASS(*), POINTER :: obj
        SELECT TYPE(obj)
        CLASS IS(t)
          PRINT *,'hello world'
        END SELECT
      END
