module mod1

  integer, parameter :: wp = selected_real_kind(13)

contains

  elemental real function func0()
    func0 = 10.0
  end function func0

  real elemental function func1()
    func1 = 10.0
  end function func1

  real(8) elemental function func2()
    func2 = 10.0_8
  end function func2

  elemental real(8) function func3()
    func3 = 10.0_8
  end function func3

  real(kind=8) elemental function func2k()
    func2k = 10.0_8
  end function func2k

  elemental real(kind=8) function func3k()
    func3k = 10.0_8
  end function func3k

  real(kind=wp) elemental function func2kwp()
    func2kwp = 10.0_wp
  end function func2kwp

  elemental real(kind=wp) function func3kwp()
    func3kwp = 10.0_wp
  end function func3kwp

  real pure function func4()
    func4 = 10.0
  end function func4

  pure real function func5()
    func5 = 10.0
  end function func5

  real(8) pure function func6()
    func6 = 10.0_8
  end function func6

  pure real(8) function func7()
    func7 = 10.0_8
  end function func7

  real(kind=8) pure function func6k()
    func6k = 10.0_8
  end function func6k

  pure real(kind=8) function func7k()
    func7k = 10.0_8
  end function func7k

  recursive real function func8(a) result(res)
    integer, intent(in) :: a
    if(a > 0) then 
      res = 10.0
    else
      res = func8(a-1)
    end if
  end function func8
 
  real recursive function func9(a) result(res)
    integer, intent(in) :: a
    if(a > 0) then 
      res = 10.0
    else
      res = func9(a-1)
    end if
  end function func9
    
  recursive real(8) function func10(a) result(res)
    integer, intent(in) :: a
    if(a > 0) then 
      res = 10.0
    else
      res = func10(a-1)
    end if
  end function func10
 
  real(8) recursive function func11(a) result(res)
    integer, intent(in) :: a
    if(a > 0) then 
      res = 10.0
    else
      res = func11(a-1)
    end if
  end function func11

  recursive real(kind=8) function func10k(a) result(res)
    integer, intent(in) :: a
    if(a > 0) then 
      res = 10.0
    else
      res = func10k(a-1)
    end if
  end function func10k
 
  real(kind=8) recursive function func11k(a) result(res)
    integer, intent(in) :: a
    if(a > 0) then 
      res = 10.0
    else
      res = func11k(a-1)
    end if
  end function func11k

  character(len=4) function func12()
    func12 = 'test'
  end function func12

end module mod1
