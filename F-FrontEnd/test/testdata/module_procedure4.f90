module mod
   private :: ss
   interface s
      module procedure ss
   end interface s
contains
   subroutine ss()
   end subroutine
end

program main
  use mod
  call s
end program main
