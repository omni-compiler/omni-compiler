      EXTERNAL :: f, g
      POINTER :: f
      f => g
      END

      SUBROUTINE s(f)
        EXTERNAL :: f, g
        POINTER :: f
        f => g
      END SUBROUTINE s

      INTEGER FUNCTION g()
      END FUNCTION g
