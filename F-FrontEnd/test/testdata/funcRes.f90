  function rfunc(x, y) result(ires)
    integer x, y
    ires = x * y
  end function rfunc

  function ifunc(x, y) result(res)
    integer x, y
    res = x + y
  end function ifunc

  real function xfunc(x, y) result(ires)
    integer x, y
    ires = x / y
  end function xfunc
