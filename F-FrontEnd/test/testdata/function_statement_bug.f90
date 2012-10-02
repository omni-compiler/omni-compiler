! this test program is for reproduction of a bug that statement_function_call()
! generate invalid intrinsic identifier('real') in compilation.
program statement_function
    real(8) :: lim01, fi
    real(8) :: r
    real(8) :: tc, times
    !
    lim01(fi)=max(0.0D0,min(1.0D0,fi))
    r = lim01(1.5D0)

    tc = 273.16
    times = min(1.e3, ( 3.56 * real(tc) + 106.7 ) * real(tc) + 1.e3 ) 
end program statement_function
