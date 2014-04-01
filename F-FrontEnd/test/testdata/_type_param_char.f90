module type_param_char
  character(4), parameter :: str = "abcd"
  integer(len_trim("abcd")) :: i
  integer(len_trim(str(1:2))) :: j
end module type_param_char
