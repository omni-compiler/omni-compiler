  MODULE m
   IMPLICIT NONE

   TYPE :: t
     INTEGER :: v
   CONTAINS
     FINAL :: missing
   END TYPE t
END MODULE m
