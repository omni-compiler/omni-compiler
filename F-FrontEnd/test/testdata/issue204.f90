PROGRAM prog1
  INTEGER :: i

  ! Comment after with continuation &
  DO i = 1, 10
    
  END DO

  !$acc data present(a,b) &
  ! Comment between sentinel ...
  !$acc present(c,d)
 
  !$acc parallel
  !$acc loop ! Comment with continuation after directive &
  ! Another comment here
  DO i = 1, 10
    
  END DO
  !$acc end parallel

  !$acc end data
END PROGRAM prog1
