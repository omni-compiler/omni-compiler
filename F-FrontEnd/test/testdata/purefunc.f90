  pure function p_funk_all_stars(george, bootsy, n) result(r)
    implicit none
    real*8, intent(in) :: george(n)
    real*8, intent(in) :: bootsy(n)
    integer, intent(in) :: n
    real*8 :: r

    integer :: i

    r = 0.0

    do i = 1, n
       r = r + george(i) * bootsy(i)
    end do

  end function p_funk_all_stars
