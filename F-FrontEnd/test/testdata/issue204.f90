PROGRAM prog1
  INTEGER :: i

  ! Comment after with continuation &
  DO i = 1, 10
    
  END DO
 
  !$acc parallel
  !$acc loop ! Comment with continuation after directive &
  DO i = 1, 10
    
  END DO
  !$acc end parallel
END PROGRAM prog1
