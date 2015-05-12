  real, allocatable :: c1(:,:),c2(:)
  allocate (c1(2,3),c2(8))
  write(*,*) "allocated(c1)=",allocated(c1)
  write(*,*) "allocated(c2)=",allocated(c2)
  deallocate (c1,c2)
  write(*,*) "allocated(c1)=",allocated(c1)
  write(*,*) "allocated(c2)=",allocated(c2)
  end
