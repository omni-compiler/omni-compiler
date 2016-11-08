program tasks

!$xmp nodes p(8)


blkname1 : block
!$xmp tasks

!$xmp task on p(1:4)
blkname2 : block
!$xmp barrier on p
end block blkname2
!$xmp end task

!$xmp task on p(5:8)
!$xmp barrier on p
!$xmp end task

!$xmp end tasks
end block blkname1

blkname3 : block
!$xmp tasks

!$xmp task on p(1:5)
blkname4 : block
!$xmp barrier on p
end block blkname4
!$xmp end task

!$xmp task on p(6:8)
blkname5 : block
!$xmp barrier on p
end block blkname5
!$xmp end task

!$xmp end tasks
end block blkname3

!$xmp task on p(1)
  write(*,*) "PASS"
!$xmp end task

end program tasks
