  module global
!!    !$xmp modules node(8)=**
    !$xmp modules node(8)
    reaal s[*]
    !$xmp coarray on node :: s
  end module global

  program image
    use global
    !$xmp tasks

    !$xmp task on node(1:4)
    call subA
    !$xmp end task

    !$xmp task on node(5:8)
    call subB
    !$xmp end task

    !$xmp end tasks
  end program image

  subroutine subA
    use global
    real, save :: a[*]

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! sync with an image outside
    !$xmp image (nodes)
    sync images(5)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! get value
    a = s[1]

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! write the value 
    write(*,*) "a=",a

  end subroutine subA

  subroutine subB
    use global
    real, save :: b[*]

    if (this_image() .eq. 1) then
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! set value
       s[1] = sqrt(2.0)
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! sync with outside images
       !$xmp image (nodes)
       sync images((/1,2,3,4/)
    end if
  end subroutine subB

