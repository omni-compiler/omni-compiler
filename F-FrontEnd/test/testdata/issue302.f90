MODULE mod1

  TYPE type_x
    INTEGER :: type
  END TYPE type_x

  TYPE type_y
    TYPE(type_x), ALLOCATABLE, DIMENSION(:) :: tx
  END TYPE type_y

  CONTAINS

  SUBROUTINE sub1()
    TYPE(type_y) :: var1
    INTEGER, PARAMETER :: TYPE1 = 1

    LOGICAL, DIMENSION(10) :: log_array

!    IF(ANY(log_array(:) == .TRUE.)) THEN 
!    END IF
    
    IF (ANY(var1%tx(1:10)%type == TYPE1) ) THEN 
    END IF
    
  END SUBROUTINE sub1


END MODULE mod1
