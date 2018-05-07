module test2mc
  integer, parameter :: lx = 204
  integer :: xlimgn
  real*8 :: abc(lx)[*]
end module test2mc
program main
  call test2
end program main

  subroutine test2
    use test2mc
    real*8 :: def(lx)

    do i=1,lx
       def(i)=real(i,8)*0.001
    enddo

    xlimgn = 1

    sync all
    if (this_image()==1) then
       abc(:)[xlimgn+1] = def(lx:1:-1)
    endif
    sync all

    nerr=0
    if (this_image()==2) then
       do i=1,lx
          if (abc(i)/=real(lx-i+1,8)*0.001) then
             nerr=nerr+1
             write(*,*) "i,def(i),abc(i)=",i,def(i),abc(i)
          endif
       enddo
    endif

    if (nerr==0) then
       print '("[",i0,"] OK")', this_image()
    else
       print '("[",i0,"] NG: nerr=",i0)', this_image(), nerr
    endif
    

    return
  end subroutine test2
