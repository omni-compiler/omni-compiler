module mo_random
implicit none
  interface random_number
     module procedure random_scalar
     module procedure random_vector
     module procedure random_array2
     module procedure random_array3
  end interface
contains
  pure subroutine random_vector(harvest)
    real, intent(out) :: harvest(:)
  end subroutine random_vector
  pure subroutine random_scalar(harvest)
    real, intent(out) :: harvest
  end subroutine random_scalar
  pure subroutine random_array2(harvest)
    real, intent(out) :: harvest(:,:)
  end subroutine random_array2
  pure subroutine random_array3(harvest)
    real, intent(out) :: harvest(:,:,:)
  end subroutine random_array3
end module mo_random
