      REAL, POINTER :: f, g, h
      EXTERNAL :: f, g
      f => g
      PRINT *, f()
      END

      SUBROUTINE s(f)
        EXTERNAL :: f, g
        POINTER :: f
        f => g
      END SUBROUTINE s

      INTEGER FUNCTION g()
      END FUNCTION g

      MODULE m
        REAL, POINTER, PRIVATE :: f
        EXTERNAL :: f
      END MODULE m
        
