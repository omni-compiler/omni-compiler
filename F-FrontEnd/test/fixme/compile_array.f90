MODULE m
    type POSVECTOR
        integer :: x,y,z
    end type
contains
FUNCTION Phase(JK,Rhat,Rarray) RESULT ( Rout )

INTEGER,                            INTENT(IN) :: JK
TYPE (POSVECTOR),                   INTENT(IN) :: Rhat    ! Far zone direction.
TYPE (POSVECTOR), DIMENSION(:,:),   INTENT(IN) :: Rarray  ! r_prime array

INTEGER, DIMENSION(size(Rarray,1),size(Rarray,2)) :: Rout

Rout = JK * (Rhat%x * Rarray%x  +  Rhat%y * Rarray%y  +  Rhat%z * Rarray%z)

END FUNCTION Phase
end module
