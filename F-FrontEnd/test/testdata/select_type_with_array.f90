      CLASS(*), POINTER :: a(:)
      SELECT TYPE(a)
      TYPE IS(INTEGER)
        PRINT *, a(1)
      END SELECT
      END
