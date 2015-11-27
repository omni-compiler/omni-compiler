  module mod1
    real a[*]
  end module mod1

  module mod2
    use mod1
    integer b(10)[*]
  end module mod2

  subroutine sub1
    use mod1
    use mod2
!!    character(4) c[*] bug354
    character(4) c(1)[*]
  end subroutine sub1

  program main1
    use mod2
    double precision d[*]
  end program main1

