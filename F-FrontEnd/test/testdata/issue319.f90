module mod1
use issue319
implicit none
contains
  

  subroutine sub2()
    if(main%id() == 10) then 
    end if
  end subroutine sub2

end module mod1
