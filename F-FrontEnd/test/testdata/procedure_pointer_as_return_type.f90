       FUNCTION f()
         PROCEDURE(),POINTER :: f
       END FUNCTION

       FUNCTION g()
         PROCEDURE(REAL),POINTER :: g
       END FUNCTION

       FUNCTION h()
         INTERFACE
           FUNCTION i()
             INTEGER :: i
           END FUNCTION i
         END INTERFACE
         PROCEDURE(i),POINTER :: h
       END FUNCTION
