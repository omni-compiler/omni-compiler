implicit none
interface
 function f()
  character(3) :: f
 end function
end interface
procedure(f), pointer :: p
contains
subroutine s()
interface
 function f()
  character(3) :: f
 end function
end interface
procedure(f), pointer :: q
q => p
end subroutine
end
