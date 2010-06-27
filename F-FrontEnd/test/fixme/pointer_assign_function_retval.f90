function a()
    integer, pointer :: a
    integer, target :: b
    b = 1
    a => b
end function
program main
    external a
    integer, pointer :: a
    integer, pointer :: b
    !b = a()
    b => a()
end program main
