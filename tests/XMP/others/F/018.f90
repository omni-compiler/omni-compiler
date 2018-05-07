!$xmp nodes p(4)
!$xmp template t(100)
!$xmp distribute t(block) onto p

      real a(100)
!$xmp align a(i) with t(i)

!$xmp loop (i) on t(i)
      do i = 1, 100
         a(i) = i
      end do

      end

