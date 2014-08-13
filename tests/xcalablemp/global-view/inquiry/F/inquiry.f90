program test_inquiry
include 'xmp_lib.h'
integer*8 xmp_desc_of
integer ichk, ierr, irank, ierrmax, error
integer xmp_node_num
integer nndims, tndims, andims
integer nsize1, nsize2, nsize3
integer index1,index2,index3
integer taxis1,taxis2,taxis3
integer aaxis1,aaxis2,aaxis3
integer offset1,offset2,offset3
logical fixed
logical replicated1,replicated2,replicated3
type(xmp_desc) dt, dt1, dn, dn1
integer lb(3),ub(3), st(3)
integer tlbound1,tlbound2,tlbound3
integer tubound1,tubound2,tubound3
integer format1,format2,format3
integer blocksize1,blocksize2,blocksize3
integer albound1,albound2,albound3
integer aubound1,aubound2,aubound3
integer lshadow1,lshadow2,lshadow3
integer ushadow1,ushadow2,ushadow3
integer map(2)
integer a(6,9,16), a1(6),iresult
integer m(2)=(/2,4/)
!$xmp nodes p(2,3,4)
!$xmp nodes p1(2)=p(1:2,1,1)
!$xmp template t(16,6,9)
!$xmp template t1(6)
!$xmp distribute t(block, cyclic, cyclic(2)) onto p
!$xmp distribute t1(gblock(m)) onto p1
!$xmp align a(i0,i1,i2) with t(i2,i0,i1)
!$xmp align a1(i0) with t1(i0)
!$xmp shadow a(1:2,2:3,3:4)

ichk=0
ierrmax=0
error=0
ierr=xmp_align_template(xmp_desc_of(a), dt)
ierrmax=ierrmax+ierr
ierr=xmp_dist_nodes(dt, dn)
ierrmax=ierrmax+ierr
ierr=xmp_align_template(xmp_desc_of(a1), dt1)
ierrmax=ierrmax+ierr
irank=xmp_node_num()

if (irank==11) then
  !ierr=xmp_nodes_eqiv(dn, dn1, lb, ub, st)
  !call check(lb(1), 1, error)
  !call check(ub(1), 2, error)
  !call check(st(1), 1, error)
  ierr=xmp_nodes_ndims(dn, nndims)
  call check(nndims, 3, error)

  ierr=xmp_nodes_index(dn, 1, index1)
  call check(index1, 1, error)
  ierr=xmp_nodes_index(dn, 2, index2)
  call check(index2, 3, error)
  ierr=xmp_nodes_index(dn, 3, index3)
  call check(index3, 2, error)

  ierr=xmp_nodes_size(dn, 1, nsize1)
  call check(nsize1, 2, error)
  ierr=xmp_nodes_size(dn, 2, nsize2)
  call check(nsize2, 3, error)
  ierr=xmp_nodes_size(dn, 3, nsize3)
  call check(nsize3, 4, error)

  !ierr=xmp_nodes_attr(dn, attr)
  !ierrmax=ierrmax+ierr
  !ierr=xmp_nodes_attr(dn1, attr)
  !ierrmax=ierrmax+ierr


  !ierr=xmp_template_fixed(dt,fixed)
  !ierrmax=ierrmax+ierr

  ierr=xmp_template_ndims(dt, tndims)
  call check(tndims, 3, error)
  ierr=xmp_template_lbound(dt, 1, tlbound1)
  call check(tlbound1, 1, error)
  ierr=xmp_template_lbound(dt, 2, tlbound2)
  call check(tlbound2, 1, error)
  ierr=xmp_template_lbound(dt, 3, tlbound3)
  call check(tlbound3, 1, error)
  ierr=xmp_template_ubound(dt, 1, tubound1)
  call check(tubound1, 16, error)
  ierr=xmp_template_ubound(dt, 2, tubound2)
  call check(tubound2, 6, error)
  ierr=xmp_template_ubound(dt, 3, tubound3)
  call check(tubound3, 9, error)
  ierr=xmp_dist_format(dt, 1, format1)
  call check(format1, 2101, error)
  ierr=xmp_dist_format(dt, 2, format2)
  call check(format2, 2102, error)
  ierr=xmp_dist_format(dt, 3, format3)
  call check(format3, 2102, error)
  ierr=xmp_dist_blocksize(dt, 1, blocksize1)
  call check(blocksize1, 8, error)
  ierr=xmp_dist_blocksize(dt, 2, blocksize2)
  call check(blocksize2, 1, error)
  ierr=xmp_dist_blocksize(dt, 3, blocksize3)
  call check(blocksize3, 2, error)
  !ierr=xmp_dist_gblockmap(dt, 1, map)
  !ierr=xmp_dist_axis(xmp_desc_of(a), 1, taxis1)
  !ierr=xmp_dist_axis(xmp_desc_of(a), 2, taxis2)
  !ierr=xmp_dist_axis(xmp_desc_of(a), 3, taxis3)
!
  !ierr=xmp_align_axis(xmp_desc_of(a), 1, aaxis1)
  !ierr=xmp_align_axis(xmp_desc_of(a), 2, aaxis2)
  !ierr=xmp_align_axis(xmp_desc_of(a), 3, aaxis3)
  ierr=xmp_align_offset(xmp_desc_of(a), 1, offset1)
  call check(offset1, 0, error)
  ierr=xmp_align_offset(xmp_desc_of(a), 2, offset2)
  call check(offset2, 0, error)
  ierr=xmp_align_offset(xmp_desc_of(a), 3, offset3)
  call check(offset3, 0, error)
  !ierr=xmp_align_replicated(xmp_desc_of(a), 1, replicated1)
  !ierr=xmp_align_replicated(xmp_desc_of(a), 2, replicated2)
  !ierr=xmp_align_replicated(xmp_desc_of(a), 3, replicated3)
  ierr=xmp_array_ndims(xmp_desc_of(a), andims)
  call check(andims, 3, error)
  ierr=xmp_array_lbound(xmp_desc_of(a), 1, albound1)
  call check(albound1, 1, error)
  ierr=xmp_array_lbound(xmp_desc_of(a), 2, albound2)
  call check(albound2, 1, error)
  ierr=xmp_array_lbound(xmp_desc_of(a), 3, albound3)
  call check(albound3, 1, error)
  ierr=xmp_array_ubound(xmp_desc_of(a), 1, aubound1)
  call check(aubound1, 6, error)
  ierr=xmp_array_ubound(xmp_desc_of(a), 2, aubound2)
  call check(aubound2, 9, error)
  ierr=xmp_array_ubound(xmp_desc_of(a), 3, aubound3)
  call check(aubound3, 16, error)
  ierr=xmp_array_lshadow(xmp_desc_of(a), 1, lshadow1)
  call check(lshadow1, 1, error)
  ierr=xmp_array_lshadow(xmp_desc_of(a), 2, lshadow2)
  call check(lshadow2, 2, error)
  ierr=xmp_array_lshadow(xmp_desc_of(a), 3, lshadow3)
  call check(lshadow3, 3, error)
  ierr=xmp_array_ushadow(xmp_desc_of(a), 1, ushadow1)
  call check(ushadow1, 2, error)
  ierr=xmp_array_ushadow(xmp_desc_of(a), 2, ushadow2)
  call check(ushadow2, 3, error)
  ierr=xmp_array_ushadow(xmp_desc_of(a), 3, ushadow3)
  call check(ushadow3, 4, error)

  if ( error .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
end if

end program test_inquiry

subroutine check(irslt,ians,error)
  integer irslt,ians,error
  !print *, 'result=',irslt,'answer',ians
  if(irslt .ne. ians) then
    error=error+1
  end if
end subroutine
