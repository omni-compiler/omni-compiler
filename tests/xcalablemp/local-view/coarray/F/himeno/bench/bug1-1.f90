  real, pointer:: p(:)
  real, target:: t(100), t2(2:99)
  p => t
  write(*,*) lbound(p,1), ubound(p,1)
  p => t2
  write(*,*) lbound(p,1), ubound(p,1)
  end
