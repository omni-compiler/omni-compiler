!! test for Ver.7

subroutine s0
  me=this_image()
  ni=num_images()
  integer s0(10)[*]
contains
  subroutine ss
    integer ss(10)[*]
    do i=1,ni
       s0(me)[i]=me
       ss(me)[i]=me
    enddo
    write(*,100) me,"s0",s0(1:ni)
    write(*,100) me,"ss",ss(1:ni)
    return
100 format("[",i0,"]",a,"=",(i0,","))
  end subroutine ss
end subroutine s0

call ss
end
