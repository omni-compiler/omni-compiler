      PROGRAM main
        use user_operation
        TYPE(t) :: a, b
        LOGICAL :: p

        a = t(1)
        b = t(1)
        a = a + b
        a = a - b
        a = a / b
        a = a * b
        a = a .HOGE. b

        p = a == b
        p = a >= b
        p = a <= b
        p = a /= b
        p = a .HUGA. b
      END PROGRAM main
