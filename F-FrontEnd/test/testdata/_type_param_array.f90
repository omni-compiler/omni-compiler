module type_param_array
  integer :: a(4) = (/1, 2, 3, 4/)
  integer(size((/1, 2, 3, 4/))) :: i
  integer(size(a)) :: j
  integer(size(a(1:1))) :: k
  integer(size((/ (i, i = 0, 2, 3) /))) :: l
end module type_param_array
