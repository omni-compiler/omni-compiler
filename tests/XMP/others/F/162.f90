! { dg-do run }
! Check that substrings behave correctly even when zero-sized
      implicit none
      character(len=10) :: s, t
      integer :: i, j

      s = "abcdefghij"
      t(:10) = s(1:)
      s(6:5) = "foo"
      if (s /= t) call exit(1)
      i = 2
      j = -1
      s(i:i+j) = "foo"
      if (s /= t) call exit(1)
      i = 20
      s(i+1:i) = "foo"
      if (s /= t) call exit(1)
      s(6:5) = s(7:5)
      if (s /= t) call exit(1)
      s = t(7:6)
      if (len(trim(s)) /= 0) call exit(1)
      if (len(t(8:4)) /= 0) call exit(1)
      if (len(trim(t(8:4))) /= 0) call exit(1)
      end
