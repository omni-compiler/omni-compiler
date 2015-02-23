!$xmp nodes p(2,2)
!$xmp template tt(20,10)
!$xmp distribute tt(block,block) onto p

  real a(2:9,2:11,18)
  real b(18,10)
!$xmp align a(*,i,j) with tt(j+2,i-1)

  real sum,max,int

  max=1.0
  int=5.0

  a=max+int
  b=0.0
!$xmp loop (i,j) on tt(j,i) reduction(+:b)
  do i=2,9
     do j=2,17
        do k=2,9
           b(j,i)=b(j,i)+a(k,i+1,j)
        enddo
     enddo
  enddo

  end
