subroutine sendp3(me)

  integer :: me(3)
  real(4), save:: buf3u(3)[*]

  buf3u(1)[me(3)-1] = 1.0

  return
end subroutine sendp3
