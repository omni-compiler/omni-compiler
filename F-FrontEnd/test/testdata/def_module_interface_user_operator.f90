      MODULE m0
        INTERFACE operator(.hoge.)
           MODULE PROCEDURE HOGE
        END INTERFACE
      CONTAINS
        FUNCTION hoge(a,b)
          INTEGER,INTENT(IN) :: a,b
          complex :: hoge
          hoge = CMPLX(b,a)
        END FUNCTION HOGE
      END MODULE m0

      PROGRAM MAIN
        use m0
        complex c
        c = 5 .hoge. 4
        print *, c
        c = 5 .HOGE. 4
        print *, c
        c = 5 . hOgE . 4
        print *, c
      END PROGRAM MAIN
