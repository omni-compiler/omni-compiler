program test 
  INTEGER :: i
  PRINT *, 'WARNING some long message ',   &
           'rest of the message'
  PRINT *, 'WARNING !!! some long message ',   &
           'rest of the message'


  ! Comment after with continuation &
  DO i = 1, 10
    
  END DO
 
  !$acc parallel
  !$acc loop ! Comment with continuation after directive &
  DO i = 1, 10
    
  END DO
  !$acc end parallel
end program test 
