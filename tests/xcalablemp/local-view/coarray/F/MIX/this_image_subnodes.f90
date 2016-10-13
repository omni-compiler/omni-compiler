      PROGRAM zzz
!!       include 'xmp_coarray.h'

!$xmp nodes p(8)
!$xmp nodes p2(2,4)=p
!$xmp nodes p3(3,2)=p(2:7)

      me=0
      me3=0

      nm=0
      nm3=0

      me=this_image()
      nm=num_images()

      nerr=0

!$xmp task on p2(1,:)
      me3=this_image()
      nm3=num_images()
      if (me3.ne.xmp_node_num()) then   !! current impementation assumed
         nerr=nerr+1
         write(*,110) "this_image()", me3,"xmp_node_num()", xmp_node_num()
      endif
      if (nm3.ne.xmp_num_nodes()) then   !! current impementation assumed
         nerr=nerr+1
         write(*,110) "num_images()", nm3,"xmp_num_nodes()", xmp_num_nodes()
      endif

110   format("[",i0,"] ERR ",a,"=",i0," while ",a,"=",i0)
!$xmp end task

      k0=0
      k3=0
      do j=1,4
         do i=1,2
            k0=k0+1
            if (i==1) then
               k3=k3+1
               if (me==k0) then
                  if (me3/=k3) then
                     nerr=nerr+1
                     write(*,100) me,"me3",k3,me3
                  endif
               end if
            end if
         end do
      end do

      if (me>0) then
         if (nm/=8) then
            nerr=nerr+1
            write(*,100) me,"nm",8,nm
         endif
      endif

      if (me2>0) then
         if (nm3/=8) then
            nerr=nerr+1
            write(*,100) me,"nm3",8,nm3
         endif
      endif

!$xmp barrier

100   format ("[",i0,"] ",a," should be ",i0," but ",i0,".")

      if (nerr==0) then 
         print '("[",i0,"] OK")', me
      else
         print '("[",i0,"] number of NGs: ",i0)', me, nerr
      end if

      END PROGRAM zzz
