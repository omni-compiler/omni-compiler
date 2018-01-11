      real function asum(a, n, a0)
!$xmp nodes p(4)
!$xmp template t(8)
!$xmp distribute t(cyclic) onto p
      real a(8)
!$xmp align a(i) with t(i)
      asum = a0
!$xmp loop on t(i)
      do i=1, n
        asum = asum+a(i)
      enddo
!$xmp reduction(+:asum)
      return
      end

!$xmp nodes p(4)
!$xmp template t(8)
!$xmp distribute t(block) onto p
      real a(8)
!$xmp align a(i) with t(i)
      real b
!$xmp loop on t(i)
      do i=1, 8
        a(i) = i
      enddo

      b = asum(a, 8, 0.0)
!$xmp task on p(1)
      if(b==36.0) then
        print *,"PASS"
      else
        print *,"ERROR"
      endif
!$xmp end task
end

