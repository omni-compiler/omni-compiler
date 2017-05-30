module named_value_module
  real, dimension(60,80,60) :: field
  integer, parameter :: i = size(field, dim=1)
  integer, parameter :: j = size(field, dim=2)
  integer, parameter :: k = size(field, dim=3)
end module named_value_module