      MODULE m3
        TYPE :: point
           REAL :: x, y
        END TYPE point

        INTERFACE
           MODULE FUNCTION point_dist(a, b) RESULT(distance)
             TYPE(POINT), INTENT(IN) :: a, b
             REAL :: distance
           END FUNCTION point_dist
        END INTERFACE
      END MODULE m3

      SUBMODULE (m3) points_a
      CONTAINS
        MODULE FUNCTION point_dist(a, b) RESULT(distance)
          TYPE(POINT), INTENT(IN) :: a, b
          REAL :: distance
          distance = sqrt((a%x - b%x)**2 + (a%y - b%y)**2)
        END FUNCTION point_dist
      END SUBMODULE points_a

      PROGRAM main
        USE m3
      END PROGRAM main
