  blockdata foo
     common /vars/x,y,z
     data x,y,z/3.,4.,5./
  end blockdata

  program user
    common /vars/a,b,c
    eps=0.00001
    if (abs(a-3.)<eps .and. abs(b-4.)<eps .and. abs(c-5.)<eps) then
       print *,"OK"
    else
       print *,"NG"
    endif
  end program user
