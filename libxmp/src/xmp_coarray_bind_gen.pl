#
# xmp_coarray_bind generator
#

for($i=1; $i<=7;$i++){
    gen_bind_subroutine("r4","real(4)",$i);
}

for($i=1; $i<=7;$i++){
    gen_bind_subroutine("r8","real(8)",$i);
}

for($i=1; $i<=7;$i++){
    gen_bind_subroutine("z8","complex(4)",$i);
}

for($i=1; $i<=7;$i++){
    gen_bind_subroutine("z16","complex(8)",$i);
}

for($i=1; $i<=7;$i++){
    gen_bind_subroutine("i2","integer(2)",$i);
}

for($i=1; $i<=7;$i++){
    gen_bind_subroutine("i4","integer(4)",$i);
}

for($i=1; $i<=7;$i++){
    gen_bind_subroutine("i8","integer(8)",$i);
}

sub gen_bind_subroutine {
    my($type,$ftype_decl,$ndims) = @_;
    local($dim_decl, $dim_decl_size,$n);
    $dim_decl = "(";
    $dim_decl_size = "(";
    $n = $ndims;
    for($i = 0; $i < $ndims; $i++){
	$dim_decl .= ":";
	$dim_decl_size .= ("l(".$n."):u(".$n.")");
	if($i != ($ndims-1)) {
	    $dim_decl .=",";
	    $dim_decl_size .=",";
	}
	$n--;
    }
    $dim_decl .= ")";
    $dim_decl_size .= ")";

    print "    subroutine xmp_coarray_bind_".$ndims."d_".$type."(desc,a_decl)\n";
    print "    integer(8) desc\n";
    print "    ".$ftype_decl.", pointer, intent(inout) :: a_decl".$dim_decl."\n";
    print "    integer(4), dimension(7) :: ub, lb\n";
    print "    integer(8) :: addr\n";
    print "    call xmp_coarray_bind_set_dim_info(desc,lb,ub,addr)\n";
    print "    call bind_cray_pointer(a_decl,lb,ub,addr)\n";
    print "    return\n";
    print "  contains\n";
    print "    subroutine bind_cray_pointer(a_decl,l,u,addr)\n";
    print "      ".$ftype_decl.", pointer, intent(inout) :: a_decl".$dim_decl."\n";
    print "      integer(4), dimension(7) :: l,u\n";
    print "      integer(8) :: addr\n";
    print "      ".$ftype_decl.":: obj".$dim_decl_size."\n";
    print "      pointer (crayptr, obj)\n";
    print "      call xmp_assign_cray_pointer(crayptr,addr)\n";
    print "      call pointer_assign(a_decl, obj)\n";
    print "    end subroutine bind_cray_pointer\n";
    print "    subroutine pointer_assign(p, d)\n";
    print "      ".$ftype_decl.", pointer :: p".$dim_decl."\n";
    print "      ".$ftype_decl.", target  :: d".$dim_decl."\n";
    print "      p => d\n";
    print "      return \n";
    print "    end subroutine pointer_assign\n";
    print "  end subroutine\n";
}
