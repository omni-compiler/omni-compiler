#
# xmp_coarray_bind generator for scalr
#
gen_bind_scalar_subroutine("r4","real(4)",4);
gen_bind_scalar_subroutine("r8","real(8)",8);
gen_bind_scalar_subroutine("z8","complex(4)",4);
gen_bind_scalar_subroutine("z16","complex(8)",8);
gen_bind_scalar_subroutine("i2","integer(2)",2);
gen_bind_scalar_subroutine("i4","integer(4)",4);
gen_bind_scalar_subroutine("i8","integer(8)",8);

sub gen_bind_scalar_subroutine {
    my($type,$ftype_decl,$type_size) = @_;

    print "    subroutine xmp_coarray_bind_".$type."(desc,v_decl)\n";
    print "    integer(8) desc\n";
    print "    ".$ftype_decl.", pointer, intent(inout) :: v_decl\n";
    print "    integer(4), dimension(7) :: ub, lb ! dummy \n";
    print "    integer(8) :: addr\n";
    print "    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)\n";
    print "    call bind_cray_pointer(v_decl,addr)\n";
    print "    return\n";
    print "  contains\n";
    print "    subroutine bind_cray_pointer(v_decl,addr)\n";
    print "      ".$ftype_decl.", pointer, intent(inout) :: v_decl\n";
    print "      integer(8) :: addr\n";
    print "      ".$ftype_decl.":: obj\n";
    print "      pointer (crayptr, obj)\n";
    print "      call xmp_assign_cray_pointer(crayptr,addr)\n";
    print "      call pointer_assign(v_decl, obj)\n";
    print "    end subroutine bind_cray_pointer\n";
    print "    subroutine pointer_assign(p, d)\n";
    print "      ".$ftype_decl.", pointer :: p\n";
    print "      ".$ftype_decl.", target  :: d\n";
    print "      p => d\n";
    print "      return \n";
    print "    end subroutine pointer_assign\n";
    print "  end subroutine\n\n";
#
    print "    subroutine xmp_coarray_mem_put_".$type."(img_dims,remote_desc,v,status)\n";
    print "    integer(8), intent(in) :: remote_desc\n";
    print "    integer(4), intent(in) :: img_dims(*)\n";
    print "    integer(4), intent(out):: status\n";
    print "    ".$ftype_decl." :: v\n";
    print "    call xmp_coarray_mem_put_addr(img_dims,remote_desc,".$type_size.",loc(v),status)\n";
    print "  end subroutine\n\n";
#
    print "    subroutine xmp_coarray_mem_get_".$type."(img_dims,remote_desc,v,status)\n";
    print "    integer(8), intent(in) :: remote_desc\n";
    print "    integer(4), intent(in) :: img_dims(*)\n";
    print "    integer(4), intent(out):: status\n";
    print "    ".$ftype_decl." :: v\n";
    print "    call xmp_coarray_mem_get_addr(img_dims,remote_desc,".$type_size.",loc(v),status)\n";
    print "  end subroutine\n\n";
}
