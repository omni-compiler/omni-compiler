      MODULE m
        public :: h
        INTERFACE
           FUNCTION f(a)
             INTEGER :: f
             INTEGER :: a
           END FUNCTION f
        END INTERFACE
        PROCEDURE(f),POINTER :: g => h
      CONTAINS
        FUNCTION h(a)
          INTEGER :: h
          INTEGER :: a
        END FUNCTION h
      END MODULE
