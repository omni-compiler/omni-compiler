module issue545_mod4
  use issue545_mod2, only: type1
  use issue545_mod3, only: type2

  implicit none 

  INTEGER, PARAMETER :: n_var = 10

  TYPE t_read_info
    CLASS(type1), POINTER :: scatter_pattern
  END TYPE
 
  TYPE t_stream_id
    TYPE(t_read_info), ALLOCATABLE :: read_info(:,:)
  END TYPE t_stream_id
 
 CONTAINS
   SUBROUTINE read_dist_INT_2D_multivar(stream_id, location)
 
     TYPE(t_stream_id), INTENT(INOUT)       :: stream_id
     INTEGER, INTENT(IN)                    :: location
 
     TYPE(type2)               :: scatter_patterns(n_var)
     INTEGER                                :: i
 
     DO i = 1, n_var
       scatter_patterns(i)%p => stream_id%read_info(location, i)%scatter_pattern
     END DO
   END SUBROUTINE read_dist_INT_2D_multivar
   
end module issue545_mod4
