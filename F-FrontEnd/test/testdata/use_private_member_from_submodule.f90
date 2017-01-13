      SUBMODULE(private_member) sub
      CONTAINS
        MODULE PROCEDURE func1
          TYPE(t), POINTER :: v
          TYPE(t), TARGET :: u
          COMPLEX :: r
          r = COMPLEX(para1, para1*10)
          v => u
          v%i = REAL(r)
          v%k = IMAG(r)
          CALL v%p1(func1)
        END PROCEDURE
      END SUBMODULE subm
