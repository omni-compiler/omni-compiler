      MODULE m
        INTEGER :: i
        BIND(c) :: i

        BIND(c) :: j
        INTEGER :: j

        INTEGER :: a
        COMMON /c1/ a
        BIND(c) :: /c1/

        INTEGER :: b
        COMMON /c2/ b
        BIND(c) :: /c2/
      END
