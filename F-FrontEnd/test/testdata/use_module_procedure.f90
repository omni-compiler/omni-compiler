      ! モジュール副プログラムの参照
      MODULE m
        USE module_function
        USE module_subroutine
        IMPLICIT NONE
        INTERFACE g
           MODULE PROCEDURE f1
        END INTERFACE
        INTERFACE sub
           MODULE PROCEDURE sub1
        END INTERFACE
      END MODULE m
