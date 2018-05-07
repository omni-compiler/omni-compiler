module mod1
integer, parameter :: fld_size = 10
  
contains

  function fct1()
    real, dimension(10) :: fct1 
  end function fct1


  real(kind=8) function fld_fr()
    dimension :: fld_fr(fld_size)
 
  end function fld_fr

end module mod1
