MODULE mod1
  
CONTAINS

  SUBROUTINE sub1
    CHARACTER (LEN=25) :: yzroutine
    REAL  ::  &
      zaa     (2)         ,& 
      zab1    (2)         ,& 
      zab2    (2)         ,& 
      zac1    (2)         ,& 
      zac2    (2)         ,& 
      zac3    (2)            

 
    yzroutine = 'sub1'

    DATA zaa      / 3.7     , 4.025   /
    DATA zab1     / 2.5648  , 3.337   /
    DATA zab2     / 1.1388  , 0.688   /
    DATA zac1     / 0.8333  , 1.285   /
    DATA zac2     / 0.2805  , 0.2305  /
    DATA zac3     / 0.1122  ,-0.1023  /

  END SUBROUTINE sub1

END MODULE mod1
