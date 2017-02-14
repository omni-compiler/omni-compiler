      INTEGER, DIMENSION(:),ALLOCATABLE:: i
      i = (/1,2,3/)
      if(i(2).eq.2) then
        PRINT *, 'PASS'
      else
        PRINT *, 'NG'
        CALL EXIT(1)
      end if
      end
