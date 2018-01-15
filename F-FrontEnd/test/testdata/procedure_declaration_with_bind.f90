       PROGRAM main
         INTERFACE
           FUNCTION f(a) BIND(C, NAME="hoge")
             integer :: f, a
           END FUNCTION f
         END INTERFACE
        CONTAINS
         SUBROUTINE s()
           PROCEDURE(f), BIND(C, NAME="hoge") :: p
         END SUBROUTINE s
       END PROGRAM main
