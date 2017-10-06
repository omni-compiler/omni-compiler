module mod1
  type t
    real, dimension(10,10) :: a
  end type t
contains

  subroutine sub1(ct)
    class(t), target, intent(in) :: ct
    real, pointer :: aptr(:) => NULL()   

    print*, ct%a(:,1)

    if(associated(aptr)) then 
      if(associated(aptr, target=ct%a(:,1))) then 
      end if
    end if

  end subroutine sub1


end module mod1
