subroutine sub
  real, target :: t(2,2)
  real, pointer :: p(:)

  p(1:4) => t(:,:)

end subroutine sub
