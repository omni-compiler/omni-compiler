module mod1
implicit none
use issue319
contains
  

  subroutine sub1()
    if(main%id() + 1 == 10) then 
    end if
  end subroutine sub1

end module mod1
