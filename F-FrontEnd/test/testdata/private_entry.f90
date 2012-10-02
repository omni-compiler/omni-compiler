module mod_private_entry
    public func
    private func_entry
  contains
  function func (a, b, c)
    integer a, b, c, func
    c = c ** 2
    entry func_entry (a, b, c) result (res)
    func = a + b + c
    res = a + b
  end function
end module mod_private_entry
