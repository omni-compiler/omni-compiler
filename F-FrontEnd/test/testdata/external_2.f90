      SUBROUTINE s1(f)
        INTEGER :: f
        INTEGER :: v
        EXTERNAL :: f
      END SUBROUTINE s1

      SUBROUTINE s2(f)
        INTEGER :: v
        EXTERNAL :: f
      END SUBROUTINE s2

      SUBROUTINE s3()
        EXTERNAL :: f
        POINTER :: f
        INTEGER :: g
        EXTERNAL :: g
        POINTER :: g
        PRINT *, f()
        PRINT *, g()
      END SUBROUTINE s3

      SUBROUTINE s4()
        EXTERNAL :: f, g
        POINTER :: f
        f => g
      END SUBROUTINE s4

