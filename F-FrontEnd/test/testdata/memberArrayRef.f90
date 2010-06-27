      PROGRAM MAIN
       TYPE tt
          INTEGER :: v(1:5)
          CHARACTER(5),DIMENSION(1:5) :: c
       END type tt

       TYPE(tt) , DIMENSION(1:5)  :: TTA

!       TTA(1:5)%v(1:5) = (/1,2,3,4,5/)
       TTA(1:5)%v(3) = (/1,2,3,4,5/)
       TTA(3)%v(1:5) = (/1,2,3,4,5/)
       TTA(3)%v(3) = 1
       TTA(1:5)%c(3)(1:5) = "abcde"
       TTA(3)%c(1:5)(1:5) = "abcde"
       TTA(3)%c(3)(1:5) = "abcde"

       print *, TTA(3)%v(3)
      END PROGRAM MAIN
