      MODULE m_to_be_read
        INTEGER, PRIVATE :: i
        INTERFACE
           MODULE FUNCTION f(a)
             REAL :: f
             REAL :: a
           END FUNCTION f
        END INTERFACE
      END MODULE m_to_be_read
