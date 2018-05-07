module mod1
contains
  function fct1(items)
    class(*), target, intent(in) :: items(:)
  end function fct1
end module mod1