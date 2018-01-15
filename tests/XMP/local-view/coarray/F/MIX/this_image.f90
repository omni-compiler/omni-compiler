      PROGRAM zzz

!$xmp nodes p(8)
!$xmp nodes p2(2,4)=p
!$xmp nodes p3(3,2)=p(2:7)

      me=0
      me2=0
      me3=0
      me5=0

      nm=0
      nm2=0
      nm3=0
      nm5=0

      me=this_image()
      nm=num_images()

!$xmp task on p2
      me2=this_image()
      nm2=num_images()
!$xmp end task

!$xmp task on p3
      me5=this_image()
      nm5=num_images()
!$xmp end task

      nerr=0

      k0=0
      k2=0
      do j=1,4
         do i=1,2
            k0=k0+1
            k2=k2+1
            if (me==k0) then
               if (me2/=k2) then
                  nerr=nerr+1
                  write(*,100) me,"me2",k2,me2
               end if
            end if
         end do
      end do

      k0=1
      k5=0
      do j=1,2
         do i=1,3
            k0=k0+1
            k5=k5+1
            if (me==k0) then
               if (me5/=k5) then
                  nerr=nerr+1
                  write(*,100) me,"me5",k5,me5
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
         if (nm2/=8) then
            nerr=nerr+1
            write(*,100) me,"nm2",8,nm2
         endif
      endif

      if (me5>0) then
         if (nm5/=6) then
            nerr=nerr+1
            write(*,100) me,"nm5",6,nm5
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
