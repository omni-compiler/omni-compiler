  integer, pointer :: ip(:,:)
  integer, target :: a(5,3)

  do j=1,3
     do i=1,5
        a(i,j) = i*10+j
     enddo
  enddo

  ip(3:,:) => a(:,2:3)

  write(*,*) ip
end program

