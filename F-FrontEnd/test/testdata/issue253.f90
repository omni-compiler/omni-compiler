program test 
  INTEGER :: i
  PRINT *, 'WARNING some long message ',   &
           'rest of the message'
  PRINT *, 'WARNING !!! some long message ',   &
           'rest of the message'
  PRINT *,'(''CAUTION !!!!! flight number '',A,'' : multi-level ODR'','' size'',I5,'' too small'')'
  PRINT *,'(''CAUTION !!!!! flight number '',A,'' : multi-level ODR'',''&
           size'',I5,'' too small'')'

  ! Comment ' jdkskjadk ' asdkljlkjd &
  DO i = 1, 10
  END DO

  ! Comment after with continuation &
  DO i = 1, 10
  END DO
 
  !$acc parallel
  !$acc loop ! Comment with continuation after directive &
  DO i = 1, 10
    
  END DO
  !$acc end parallel

  !$omp loop ! jsahdjkhd  &
  DO i = 1, 10
  END DO


  !$xmp node ! jsahdjkhd  &
  DO i = 1, 10
  END DO


  !$claw nodep ! jsahdjkhd  &
  DO i = 1, 10
  END DO
end program test 
