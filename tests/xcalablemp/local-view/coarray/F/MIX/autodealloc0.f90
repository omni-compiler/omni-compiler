  !$xmp nodes p(8)
  !$xmp nodes q1(3)=p(1:3)
  !$xmp nodes q2(2,2)=p(4:7)

  real as(10,10)[*]
  real,allocatable:: ad(:,:)[:]

  call nfoo(as,ad)
  write(*,*) "allocated_bytes= ", xmpf_coarray_allocated_bytes()
  write(*,*) "garbage_bytes  = ", xmpf_coarray_garbage_bytes()

  contains

    subroutine nfoo(s,d)
      real s(10,10)
      real, allocatable :: d(:,:)[:]

      allocate (d(10,10)[*])

      return
    end subroutine nfoo

  end 


