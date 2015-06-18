program main
  real, pointer:: p(:)
  real, target:: t(100)
  integer n

  n = 2
  call sub(p, t)
  write(*,*) lbound(p,1), ubound(p,1)

contains
  subroutine sub(pp, tt)
    real, pointer:: pp(:)
    real,target:: tt(n:)
    pp => tt
    return
  end subroutine sub

end program main
