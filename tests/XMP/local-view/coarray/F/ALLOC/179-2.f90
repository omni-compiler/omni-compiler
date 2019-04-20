program test
 real(8),allocatable :: A(:,:)[:]
 real(8) :: B(3,5)

 allocate(A(3,5)[*])
 A(:,:) = 0.d0
 B(:,:) = 0.d0
 A(:,1)[1] = B(:,1)
end program test
