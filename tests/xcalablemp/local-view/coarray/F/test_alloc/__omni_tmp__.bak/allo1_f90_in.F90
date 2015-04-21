SUBROUTINE xmpf_main ( )
 EXTERNAL xmpf_coarray_proc_init
 INTEGER ( KIND= 8 ) :: xmpf_resource_tag
# 4 "allo1.f90"
 REAL , ALLOCATABLE :: b ( : )
# 3 "allo1.f90"
 REAL , POINTER :: a ( : )
# 32 "xmp_lib.h"
 INTEGER :: xmpf_coarray_image
 EXTERNAL xmpf_coarray_image
# 17 "xmp_lib.h"
 INTEGER :: this_image
 EXTERNAL this_image
 INTEGER :: nerr
 INTEGER :: xmp_size_array ( 0 : 15 , 0 : 6 )
 INTEGER ( KIND= 8 ) :: xmpf_descptr_a
# 17 "xmp_lib.h"
 INTEGER :: num_images
 EXTERNAL num_images
 INTEGER :: me
 EXTERNAL xmpf_coarray_set_coshape
 EXTERNAL xmpf_coarray_proc_finalize
# 4 "xmp_lib.h"
 TYPE :: xmp_desc
  SEQUENCE
  INTEGER ( KIND= 8 ) :: desc
 END TYPE xmp_desc
# 4 "xmp_lib_coarray_sync.h"
 INTERFACE xmpf_sync_all
# 5 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_all_nostat ( )
  END SUBROUTINE xmpf_sync_all_nostat
# 7 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_all_stat_wrap ( stat , errmsg )
# 8 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(OUT) :: stat
# 9 "xmp_lib_coarray_sync.h"
   CHARACTER ( LEN= * ) , OPTIONAL , INTENT(OUT) :: errmsg
  END SUBROUTINE xmpf_sync_all_stat_wrap
 END INTERFACE
# 16 "xmp_lib_coarray_sync.h"
 INTERFACE xmpf_sync_memory
# 17 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_memory_nostat ( )
  END SUBROUTINE xmpf_sync_memory_nostat
# 19 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_memory_stat_wrap ( stat , errmsg )
# 20 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(OUT) :: stat
# 21 "xmp_lib_coarray_sync.h"
   CHARACTER ( LEN= * ) , OPTIONAL , INTENT(OUT) :: errmsg
  END SUBROUTINE xmpf_sync_memory_stat_wrap
 END INTERFACE
# 28 "xmp_lib_coarray_sync.h"
 INTERFACE xmpf_sync_images
# 29 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_image_nostat ( image )
# 30 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(IN) :: image
  END SUBROUTINE xmpf_sync_image_nostat
# 32 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_images_nostat_wrap ( images )
# 33 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(IN) :: images ( : )
  END SUBROUTINE xmpf_sync_images_nostat_wrap
# 35 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_allimages_nostat_wrap ( aster )
# 36 "xmp_lib_coarray_sync.h"
   CHARACTER , INTENT(IN) :: aster
  END SUBROUTINE xmpf_sync_allimages_nostat_wrap
# 39 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_image_stat_wrap ( image , stat , errmsg )
# 40 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(IN) :: image
# 41 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(OUT) :: stat
# 42 "xmp_lib_coarray_sync.h"
   CHARACTER ( LEN= * ) , OPTIONAL , INTENT(OUT) :: errmsg
  END SUBROUTINE xmpf_sync_image_stat_wrap
# 44 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_images_stat_wrap ( images , stat , errmsg )
# 45 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(IN) :: images ( : )
# 46 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(OUT) :: stat
# 47 "xmp_lib_coarray_sync.h"
   CHARACTER ( LEN= * ) , OPTIONAL , INTENT(OUT) :: errmsg
  END SUBROUTINE xmpf_sync_images_stat_wrap
# 49 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_sync_allimages_stat_wrap ( aster , stat , errmsg )
# 50 "xmp_lib_coarray_sync.h"
   CHARACTER , INTENT(IN) :: aster
# 51 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(OUT) :: stat
# 52 "xmp_lib_coarray_sync.h"
   CHARACTER ( LEN= * ) , OPTIONAL , INTENT(OUT) :: errmsg
  END SUBROUTINE xmpf_sync_allimages_stat_wrap
 END INTERFACE
# 60 "xmp_lib_coarray_sync.h"
 INTERFACE
# 61 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_lock ( stat , errmsg )
# 62 "xmp_lib_coarray_sync.h"
   INTEGER , OPTIONAL , INTENT(OUT) :: stat
# 63 "xmp_lib_coarray_sync.h"
   CHARACTER ( LEN= * ) , OPTIONAL , INTENT(OUT) :: errmsg
  END SUBROUTINE xmpf_lock
 END INTERFACE
# 68 "xmp_lib_coarray_sync.h"
 INTERFACE
# 69 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_unlock ( stat , errmsg )
# 70 "xmp_lib_coarray_sync.h"
   INTEGER , OPTIONAL , INTENT(OUT) :: stat
# 71 "xmp_lib_coarray_sync.h"
   CHARACTER ( LEN= * ) , OPTIONAL , INTENT(OUT) :: errmsg
  END SUBROUTINE xmpf_unlock
 END INTERFACE
# 78 "xmp_lib_coarray_sync.h"
 INTERFACE
# 79 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_critical ( )
  END SUBROUTINE xmpf_critical
 END INTERFACE
# 83 "xmp_lib_coarray_sync.h"
 INTERFACE
# 84 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_end_critical ( )
  END SUBROUTINE xmpf_end_critical
 END INTERFACE
# 91 "xmp_lib_coarray_sync.h"
 INTERFACE
# 92 "xmp_lib_coarray_sync.h"
  SUBROUTINE xmpf_error_stop ( )
  END SUBROUTINE xmpf_error_stop
 END INTERFACE
# 103 "xmp_lib_coarray_sync.h"
 INTERFACE atmic_define
# 104 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_define_i2 ( atom , value )
# 105 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(OUT) :: atom
# 106 "xmp_lib_coarray_sync.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: value
  END SUBROUTINE atomic_define_i2
# 108 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_define_i4 ( atom , value )
# 109 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(OUT) :: atom
# 110 "xmp_lib_coarray_sync.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: value
  END SUBROUTINE atomic_define_i4
# 112 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_define_i8 ( atom , value )
# 113 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(OUT) :: atom
# 114 "xmp_lib_coarray_sync.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: value
  END SUBROUTINE atomic_define_i8
# 116 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_define_l2 ( atom , value )
# 117 "xmp_lib_coarray_sync.h"
   LOGICAL , INTENT(OUT) :: atom
# 118 "xmp_lib_coarray_sync.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: value
  END SUBROUTINE atomic_define_l2
# 120 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_define_l4 ( atom , value )
# 121 "xmp_lib_coarray_sync.h"
   LOGICAL , INTENT(OUT) :: atom
# 122 "xmp_lib_coarray_sync.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: value
  END SUBROUTINE atomic_define_l4
# 124 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_define_l8 ( atom , value )
# 125 "xmp_lib_coarray_sync.h"
   LOGICAL , INTENT(OUT) :: atom
# 126 "xmp_lib_coarray_sync.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: value
  END SUBROUTINE atomic_define_l8
 END INTERFACE
# 130 "xmp_lib_coarray_sync.h"
 INTERFACE atmic_ref
# 131 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_ref_i2 ( value , atom )
# 132 "xmp_lib_coarray_sync.h"
   INTEGER ( KIND= 2 ) , INTENT(OUT) :: value
# 133 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(IN) :: atom
  END SUBROUTINE atomic_ref_i2
# 135 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_ref_i4 ( value , atom )
# 136 "xmp_lib_coarray_sync.h"
   INTEGER ( KIND= 4 ) , INTENT(OUT) :: value
# 137 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(IN) :: atom
  END SUBROUTINE atomic_ref_i4
# 139 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_ref_i8 ( value , atom )
# 140 "xmp_lib_coarray_sync.h"
   INTEGER ( KIND= 8 ) , INTENT(OUT) :: value
# 141 "xmp_lib_coarray_sync.h"
   INTEGER , INTENT(IN) :: atom
  END SUBROUTINE atomic_ref_i8
# 143 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_ref_l2 ( value , atom )
# 144 "xmp_lib_coarray_sync.h"
   LOGICAL ( KIND= 2 ) , INTENT(OUT) :: value
# 145 "xmp_lib_coarray_sync.h"
   LOGICAL , INTENT(IN) :: atom
  END SUBROUTINE atomic_ref_l2
# 147 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_ref_l4 ( value , atom )
# 148 "xmp_lib_coarray_sync.h"
   LOGICAL ( KIND= 4 ) , INTENT(OUT) :: value
# 149 "xmp_lib_coarray_sync.h"
   LOGICAL , INTENT(IN) :: atom
  END SUBROUTINE atomic_ref_l4
# 151 "xmp_lib_coarray_sync.h"
  SUBROUTINE atomic_ref_l8 ( value , atom )
# 152 "xmp_lib_coarray_sync.h"
   LOGICAL ( KIND= 8 ) , INTENT(OUT) :: value
# 153 "xmp_lib_coarray_sync.h"
   LOGICAL , INTENT(IN) :: atom
  END SUBROUTINE atomic_ref_l8
 END INTERFACE
# 4 "xmp_lib_coarray_get.h"
 INTERFACE xmpf_coarray_get0d
# 9 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_i2 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 15 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) :: val
# 12 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 13 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 13 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 13 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 14 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_i2
# 17 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_i4 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 23 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) :: val
# 20 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 21 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 21 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 21 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 22 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_i4
# 25 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_i8 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 31 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) :: val
# 28 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 29 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 29 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 29 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 30 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_i8
# 33 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_l2 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 39 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) :: val
# 36 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 37 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 37 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 37 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 38 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_l2
# 41 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_l4 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 47 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) :: val
# 44 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 45 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 45 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 45 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 46 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_l4
# 49 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_l8 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 55 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) :: val
# 52 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 53 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 53 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 53 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 54 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_l8
# 57 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_r4 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 63 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) :: val
# 60 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 61 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 61 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 61 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 62 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_r4
# 65 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_r8 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 71 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) :: val
# 68 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 69 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 69 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 69 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 70 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_r8
# 73 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_z8 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 79 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) :: val
# 76 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 77 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 77 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 77 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 78 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_z8
# 81 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_z16 ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 87 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) :: val
# 84 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 85 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 85 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 85 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 86 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: baseaddr
  END FUNCTION xmpf_coarray_get0d_z16
# 89 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get0d_cn ( descptr , baseaddr , element , coindex , rank ) RESULT(val)
# 92 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 93 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 93 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 93 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 94 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: baseaddr
# 95 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) :: val
  END FUNCTION xmpf_coarray_get0d_cn
 END INTERFACE
# 101 "xmp_lib_coarray_get.h"
 INTERFACE xmpf_coarray_get1d
# 106 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_i2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 110 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 111 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 111 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 111 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 112 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 113 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 114 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 115 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_i2
# 117 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_i4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 121 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 122 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 122 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 122 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 123 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 124 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 125 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 126 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_i4
# 128 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_i8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 132 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 133 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 133 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 133 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 134 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 135 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 136 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 137 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_i8
# 139 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_l2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 143 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 144 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 144 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 144 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 145 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 146 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 147 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 148 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_l2
# 150 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_l4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 154 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 155 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 155 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 155 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 156 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 157 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 158 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 159 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_l4
# 161 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_l8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 165 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 166 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 166 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 166 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 167 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 168 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 169 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 170 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_l8
# 172 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_r4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 176 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 177 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 177 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 177 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 178 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 179 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 180 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 181 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_r4
# 183 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_r8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 187 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 188 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 188 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 188 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 189 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 190 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 191 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 192 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_r8
# 194 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_z8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 198 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 199 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 199 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 199 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 200 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 201 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 202 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 203 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_z8
# 205 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_z16 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 209 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 210 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 210 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 210 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 211 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 212 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 213 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 214 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_z16
# 216 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get1d_cn ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 ) RESULT(val)
# 220 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 221 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 221 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 221 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 222 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 223 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: baseaddr
# 224 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr1
# 225 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) :: val ( 1 : count1 )
  END FUNCTION xmpf_coarray_get1d_cn
 END INTERFACE
# 231 "xmp_lib_coarray_get.h"
 INTERFACE xmpf_coarray_get2d
# 236 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_i2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 241 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 242 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 242 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 242 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 243 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 244 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 245 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 246 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 247 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 248 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_i2
# 251 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_i4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 256 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 257 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 257 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 257 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 258 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 259 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 260 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 261 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 262 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 263 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_i4
# 266 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_i8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 271 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 272 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 272 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 272 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 273 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 274 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 275 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 276 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 277 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 278 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_i8
# 281 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_l2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 286 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 287 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 287 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 287 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 288 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 289 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 290 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 291 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 292 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 293 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_l2
# 296 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_l4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 301 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 302 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 302 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 302 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 303 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 304 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 305 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 306 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 307 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 308 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_l4
# 311 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_l8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 316 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 317 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 317 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 317 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 318 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 319 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 320 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 321 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 322 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 323 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_l8
# 326 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_r4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 331 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 332 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 332 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 332 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 333 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 334 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 335 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 336 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 337 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 338 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_r4
# 341 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_r8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 346 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 347 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 347 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 347 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 348 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 349 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 350 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 351 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 352 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 353 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_r8
# 356 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_z8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 361 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 362 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 362 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 362 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 363 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 364 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 365 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 366 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 367 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 368 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_z8
# 371 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_z16 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 376 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 377 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 377 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 377 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 378 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 379 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 380 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 381 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 382 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 383 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_z16
# 386 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get2d_cn ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 )&
   RESULT(val)
# 391 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 392 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 392 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 392 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 393 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 394 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 395 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: baseaddr
# 396 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr1
# 397 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr2
# 398 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) :: val ( 1 : count1 , 1 : count2 )
  END FUNCTION xmpf_coarray_get2d_cn
 END INTERFACE
# 405 "xmp_lib_coarray_get.h"
 INTERFACE xmpf_coarray_get3d
# 410 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_i2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 416 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 417 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 417 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 417 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 418 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 419 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 420 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 421 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 422 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 423 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 424 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 425 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_i2
# 428 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_i4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 434 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 435 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 435 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 435 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 436 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 437 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 438 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 439 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 440 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 441 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 442 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 443 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_i4
# 446 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_i8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 452 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 453 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 453 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 453 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 454 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 455 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 456 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 457 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 458 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 459 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 460 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 461 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_i8
# 464 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_l2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 470 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 471 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 471 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 471 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 472 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 473 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 474 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 475 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 476 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 477 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 478 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 479 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_l2
# 482 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_l4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 488 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 489 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 489 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 489 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 490 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 491 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 492 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 493 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 494 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 495 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 496 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 497 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_l4
# 500 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_l8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 506 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 507 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 507 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 507 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 508 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 509 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 510 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 511 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 512 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 513 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 514 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 515 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_l8
# 518 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_r4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 524 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 525 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 525 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 525 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 526 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 527 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 528 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 529 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 530 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 531 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 532 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 533 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_r4
# 536 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_r8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 542 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 543 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 543 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 543 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 544 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 545 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 546 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 547 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 548 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 549 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 550 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 551 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_r8
# 554 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_z8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 560 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 561 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 561 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 561 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 562 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 563 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 564 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 565 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 566 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 567 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 568 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 569 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_z8
# 572 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_z16 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 578 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 579 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 579 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 579 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 580 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 581 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 582 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 583 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 584 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 585 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 586 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 587 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_z16
# 590 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get3d_cn ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 ) RESULT(val)
# 596 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 597 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 597 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 597 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 598 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 599 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 600 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 601 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: baseaddr
# 602 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr1
# 603 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr2
# 604 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr3
# 605 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 )
  END FUNCTION xmpf_coarray_get3d_cn
 END INTERFACE
# 612 "xmp_lib_coarray_get.h"
 INTERFACE xmpf_coarray_get4d
# 617 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_i2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 624 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 625 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 625 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 625 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 626 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 627 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 628 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 629 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 630 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 631 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 632 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 633 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 634 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr4
# 635 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_i2
# 638 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_i4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 645 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 646 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 646 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 646 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 647 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 648 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 649 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 650 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 651 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 652 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 653 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 654 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 655 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 656 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_i4
# 659 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_i8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 666 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 667 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 667 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 667 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 668 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 669 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 670 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 671 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 672 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 673 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 674 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 675 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 676 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 677 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_i8
# 680 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_l2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 687 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 688 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 688 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 688 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 689 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 690 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 691 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 692 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 693 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 694 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 695 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 696 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 697 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr4
# 698 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_l2
# 701 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_l4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 708 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 709 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 709 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 709 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 710 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 711 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 712 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 713 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 714 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 715 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 716 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 717 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 718 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 719 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_l4
# 722 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_l8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 729 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 730 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 730 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 730 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 731 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 732 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 733 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 734 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 735 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 736 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 737 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 738 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 739 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 740 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_l8
# 743 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_r4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 750 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 751 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 751 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 751 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 752 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 753 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 754 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 755 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 756 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 757 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 758 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 759 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 760 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 761 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_r4
# 764 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_r8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 771 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 772 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 772 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 772 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 773 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 774 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 775 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 776 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 777 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 778 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 779 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 780 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 781 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 782 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_r8
# 785 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_z8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 792 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 793 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 793 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 793 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 794 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 795 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 796 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 797 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 798 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 799 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 800 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 801 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 802 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 803 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_z8
# 806 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_z16 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 813 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 814 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 814 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 814 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 815 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 816 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 817 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 818 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 819 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 820 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 821 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 822 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 823 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 824 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_z16
# 827 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get4d_cn ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 ) RESULT(val)
# 834 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 835 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 835 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 835 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 836 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 837 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 838 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 839 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 840 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: baseaddr
# 841 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr1
# 842 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr2
# 843 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr3
# 844 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr4
# 845 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 )
  END FUNCTION xmpf_coarray_get4d_cn
 END INTERFACE
# 852 "xmp_lib_coarray_get.h"
 INTERFACE xmpf_coarray_get5d
# 857 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_i2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 865 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 866 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 866 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 866 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 867 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 868 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 869 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 870 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 871 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 872 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 873 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 874 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 875 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 876 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr4
# 877 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr5
# 878 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_i2
# 881 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_i4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 889 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 890 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 890 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 890 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 891 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 892 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 893 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 894 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 895 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 896 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 897 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 898 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 899 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 900 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 901 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 902 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_i4
# 905 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_i8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 913 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 914 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 914 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 914 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 915 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 916 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 917 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 918 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 919 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 920 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 921 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 922 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 923 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 924 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 925 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 926 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_i8
# 929 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_l2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 937 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 938 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 938 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 938 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 939 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 940 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 941 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 942 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 943 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 944 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 945 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 946 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 947 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 948 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr4
# 949 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr5
# 950 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_l2
# 953 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_l4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 961 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 962 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 962 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 962 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 963 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 964 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 965 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 966 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 967 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 968 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 969 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 970 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 971 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 972 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 973 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 974 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_l4
# 977 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_l8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 985 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 986 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 986 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 986 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 987 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 988 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 989 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 990 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 991 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 992 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 993 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 994 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 995 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 996 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 997 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 998 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_l8
# 1001 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_r4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 1009 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1010 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1010 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1010 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1011 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1012 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1013 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1014 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1015 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1016 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1017 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1018 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1019 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1020 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1021 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1022 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_r4
# 1025 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_r8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 1033 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1034 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1034 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1034 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1035 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1036 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1037 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1038 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1039 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1040 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1041 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1042 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1043 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1044 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1045 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1046 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_r8
# 1049 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_z8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 1057 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1058 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1058 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1058 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1059 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1060 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1061 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1062 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1063 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1064 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1065 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1066 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1067 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1068 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1069 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1070 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_z8
# 1073 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_z16 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 1081 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1082 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1082 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1082 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1083 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1084 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1085 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1086 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1087 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1088 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1089 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1090 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1091 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1092 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1093 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1094 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_z16
# 1097 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get5d_cn ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 ) RESULT(val)
# 1105 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1106 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1106 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1106 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1107 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1108 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1109 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1110 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1111 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1112 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: baseaddr
# 1113 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr1
# 1114 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr2
# 1115 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr3
# 1116 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr4
# 1117 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr5
# 1118 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 )
  END FUNCTION xmpf_coarray_get5d_cn
 END INTERFACE
# 1125 "xmp_lib_coarray_get.h"
 INTERFACE xmpf_coarray_get6d
# 1130 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_i2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1139 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1140 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1140 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1140 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1141 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1142 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1143 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1144 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1145 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1146 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1147 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 1148 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 1149 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 1150 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 1151 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr4
# 1152 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr5
# 1153 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr6
# 1154 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_i2
# 1157 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_i4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1166 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1167 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1167 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1167 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1168 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1169 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1170 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1171 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1172 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1173 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1174 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1175 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1176 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1177 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1178 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1179 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1180 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr6
# 1181 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_i4
# 1184 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_i8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1193 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1194 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1194 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1194 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1195 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1196 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1197 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1198 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1199 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1200 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1201 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1202 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1203 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1204 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1205 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1206 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1207 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr6
# 1208 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_i8
# 1211 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_l2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1220 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1221 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1221 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1221 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1222 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1223 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1224 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1225 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1226 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1227 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1228 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 1229 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 1230 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 1231 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 1232 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr4
# 1233 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr5
# 1234 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr6
# 1235 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_l2
# 1238 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_l4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1247 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1248 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1248 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1248 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1249 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1250 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1251 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1252 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1253 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1254 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1255 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1256 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1257 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1258 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1259 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1260 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1261 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr6
# 1262 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_l4
# 1265 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_l8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1274 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1275 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1275 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1275 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1276 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1277 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1278 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1279 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1280 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1281 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1282 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1283 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1284 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1285 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1286 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1287 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1288 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr6
# 1289 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_l8
# 1292 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_r4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1301 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1302 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1302 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1302 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1303 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1304 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1305 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1306 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1307 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1308 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1309 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1310 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1311 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1312 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1313 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1314 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1315 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr6
# 1316 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_r4
# 1319 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_r8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1328 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1329 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1329 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1329 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1330 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1331 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1332 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1333 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1334 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1335 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1336 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1337 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1338 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1339 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1340 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1341 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1342 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr6
# 1343 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_r8
# 1346 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_z8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1355 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1356 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1356 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1356 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1357 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1358 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1359 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1360 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1361 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1362 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1363 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1364 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1365 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1366 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1367 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1368 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1369 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr6
# 1370 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_z8
# 1373 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_z16 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1382 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1383 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1383 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1383 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1384 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1385 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1386 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1387 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1388 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1389 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1390 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1391 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1392 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1393 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1394 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1395 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1396 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr6
# 1397 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_z16
# 1400 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get6d_cn ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 ) RESULT(val)
# 1409 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1410 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1410 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1410 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1411 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1412 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1413 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1414 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1415 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1416 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1417 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: baseaddr
# 1418 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr1
# 1419 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr2
# 1420 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr3
# 1421 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr4
# 1422 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr5
# 1423 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr6
# 1424 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 )
  END FUNCTION xmpf_coarray_get6d_cn
 END INTERFACE
# 1431 "xmp_lib_coarray_get.h"
 INTERFACE xmpf_coarray_get7d
# 1436 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_i2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1446 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1447 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1447 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1447 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1448 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1449 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1450 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1451 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1452 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1453 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1454 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1455 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 1456 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 1457 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 1458 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 1459 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr4
# 1460 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr5
# 1461 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr6
# 1462 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) , INTENT(IN) :: nextaddr7
# 1463 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_i2
# 1466 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_i4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1476 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1477 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1477 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1477 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1478 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1479 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1480 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1481 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1482 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1483 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1484 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1485 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1486 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1487 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1488 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1489 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1490 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1491 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr6
# 1492 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) , INTENT(IN) :: nextaddr7
# 1493 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_i4
# 1496 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_i8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1506 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1507 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1507 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1507 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1508 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1509 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1510 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1511 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1512 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1513 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1514 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1515 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1516 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1517 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1518 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1519 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1520 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1521 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr6
# 1522 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: nextaddr7
# 1523 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_i8
# 1526 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_l2 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1536 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1537 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1537 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1537 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1538 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1539 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1540 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1541 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1542 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1543 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1544 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1545 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: baseaddr
# 1546 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr1
# 1547 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr2
# 1548 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr3
# 1549 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr4
# 1550 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr5
# 1551 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr6
# 1552 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) , INTENT(IN) :: nextaddr7
# 1553 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 2 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_l2
# 1556 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_l4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1566 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1567 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1567 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1567 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1568 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1569 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1570 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1571 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1572 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1573 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1574 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1575 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1576 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1577 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1578 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1579 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1580 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1581 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr6
# 1582 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) , INTENT(IN) :: nextaddr7
# 1583 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_l4
# 1586 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_l8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1596 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1597 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1597 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1597 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1598 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1599 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1600 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1601 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1602 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1603 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1604 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1605 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1606 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1607 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1608 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1609 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1610 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1611 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr6
# 1612 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) , INTENT(IN) :: nextaddr7
# 1613 "xmp_lib_coarray_get.h"
   LOGICAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_l8
# 1616 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_r4 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1626 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1627 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1627 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1627 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1628 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1629 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1630 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1631 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1632 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1633 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1634 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1635 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1636 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1637 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1638 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1639 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1640 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1641 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr6
# 1642 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) , INTENT(IN) :: nextaddr7
# 1643 "xmp_lib_coarray_get.h"
   REAL ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_r4
# 1646 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_r8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1656 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1657 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1657 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1657 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1658 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1659 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1660 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1661 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1662 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1663 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1664 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1665 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1666 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1667 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1668 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1669 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1670 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1671 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr6
# 1672 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) , INTENT(IN) :: nextaddr7
# 1673 "xmp_lib_coarray_get.h"
   REAL ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_r8
# 1676 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_z8 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1686 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1687 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1687 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1687 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1688 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1689 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1690 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1691 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1692 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1693 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1694 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1695 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: baseaddr
# 1696 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr1
# 1697 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr2
# 1698 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr3
# 1699 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr4
# 1700 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr5
# 1701 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr6
# 1702 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) , INTENT(IN) :: nextaddr7
# 1703 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 4 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_z8
# 1706 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_z16 ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1716 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1717 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1717 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1717 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1718 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1719 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1720 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1721 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1722 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1723 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1724 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1725 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: baseaddr
# 1726 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr1
# 1727 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr2
# 1728 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr3
# 1729 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr4
# 1730 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr5
# 1731 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr6
# 1732 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) , INTENT(IN) :: nextaddr7
# 1733 "xmp_lib_coarray_get.h"
   COMPLEX ( KIND= 8 ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_z16
# 1736 "xmp_lib_coarray_get.h"
  FUNCTION xmpf_coarray_get7d_cn ( descptr , baseaddr , element , coindex , rank , nextaddr1 , count1 , nextaddr2 , count2 ,&
   nextaddr3 , count3 , nextaddr4 , count4 , nextaddr5 , count5 , nextaddr6 , count6 , nextaddr7 , count7 ) RESULT(val)
# 1746 "xmp_lib_coarray_get.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1747 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: element
# 1747 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: coindex
# 1747 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: rank
# 1748 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count1
# 1749 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count2
# 1750 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count3
# 1751 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count4
# 1752 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count5
# 1753 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count6
# 1754 "xmp_lib_coarray_get.h"
   INTEGER , INTENT(IN) :: count7
# 1755 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: baseaddr
# 1756 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr1
# 1757 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr2
# 1758 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr3
# 1759 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr4
# 1760 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr5
# 1761 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr6
# 1762 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) , INTENT(IN) :: nextaddr7
# 1763 "xmp_lib_coarray_get.h"
   CHARACTER ( LEN= element ) :: val ( 1 : count1 , 1 : count2 , 1 : count3 , 1 : count4 , 1 : count5 , 1 : count6 , 1 : count7 )
  END FUNCTION xmpf_coarray_get7d_cn
 END INTERFACE
# 4 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_alloc0d
# 9 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_i2 ( descptr , var , count , element , tag , rank )
# 11 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 12 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 13 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 13 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 13 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 14 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_i2
# 17 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_i4 ( descptr , var , count , element , tag , rank )
# 19 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 20 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 21 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 21 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 21 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 22 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_i4
# 25 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_i8 ( descptr , var , count , element , tag , rank )
# 27 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 28 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 29 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 29 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 29 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 30 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_i8
# 33 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_l2 ( descptr , var , count , element , tag , rank )
# 35 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 36 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 37 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 37 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 37 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 38 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_l2
# 41 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_l4 ( descptr , var , count , element , tag , rank )
# 43 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 44 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 45 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 45 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 45 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 46 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_l4
# 49 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_l8 ( descptr , var , count , element , tag , rank )
# 51 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 52 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 53 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 53 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 53 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 54 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_l8
# 57 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_r4 ( descptr , var , count , element , tag , rank )
# 59 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 60 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 61 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 61 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 61 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 62 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_r4
# 65 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_r8 ( descptr , var , count , element , tag , rank )
# 67 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 68 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 69 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 69 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 69 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 70 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_r8
# 73 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_z8 ( descptr , var , count , element , tag , rank )
# 75 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 76 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 77 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 77 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 77 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 78 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_z8
# 81 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_z16 ( descptr , var , count , element , tag , rank )
# 83 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 84 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 85 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 85 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 85 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 86 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_z16
# 89 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc0d_cn ( descptr , var , count , element , tag , rank )
# 91 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 92 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 93 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 93 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 93 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 94 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= element ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_alloc0d_cn
 END INTERFACE
# 100 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_alloc1d
# 105 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_i2 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 107 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 108 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 109 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 109 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 109 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 109 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 109 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 110 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_i2
# 113 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_i4 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 115 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 116 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 117 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 117 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 117 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 117 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 117 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 118 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_i4
# 121 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_i8 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 123 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 124 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 125 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 125 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 125 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 125 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 125 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 126 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_i8
# 129 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_l2 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 131 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 132 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 133 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 133 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 133 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 133 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 133 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 134 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_l2
# 137 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_l4 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 139 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 140 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 141 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 141 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 141 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 141 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 141 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 142 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_l4
# 145 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_l8 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 147 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 148 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 149 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 149 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 149 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 149 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 149 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 150 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_l8
# 153 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_r4 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 155 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 156 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 157 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 157 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 157 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 157 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 157 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 158 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_r4
# 161 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_r8 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 163 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 164 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 165 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 165 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 165 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 165 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 165 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 166 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_r8
# 169 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_z8 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 171 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 172 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 173 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 173 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 173 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 173 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 173 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 174 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_z8
# 177 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_z16 ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 179 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 180 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 181 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 181 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 181 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 181 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 181 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 182 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_z16
# 185 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc1d_cn ( descptr , var , count , element , tag , rank , lb1 , ub1 )
# 187 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 188 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 189 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 189 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 189 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 189 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 189 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 190 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= element ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_alloc1d_cn
 END INTERFACE
# 196 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_alloc2d
# 201 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_i2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 203 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 204 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 205 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 205 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 205 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 205 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 205 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 205 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 205 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 206 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_i2
# 209 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_i4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 211 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 212 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 213 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 213 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 213 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 213 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 213 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 213 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 213 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 214 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_i4
# 217 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_i8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 219 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 220 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 221 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 221 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 221 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 221 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 221 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 221 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 221 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 222 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_i8
# 225 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_l2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 227 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 228 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 229 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 229 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 229 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 229 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 229 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 229 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 229 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 230 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_l2
# 233 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_l4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 235 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 236 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 237 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 237 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 237 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 237 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 237 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 237 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 237 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 238 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_l4
# 241 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_l8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 243 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 244 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 245 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 245 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 245 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 245 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 245 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 245 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 245 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 246 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_l8
# 249 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_r4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 251 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 252 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 253 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 253 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 253 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 253 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 253 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 253 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 253 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 254 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_r4
# 257 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_r8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 259 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 260 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 261 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 261 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 261 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 261 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 261 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 261 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 261 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 262 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_r8
# 265 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_z8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 267 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 268 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 269 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 269 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 269 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 269 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 269 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 269 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 269 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 270 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_z8
# 273 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_z16 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 275 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 276 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 277 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 277 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 277 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 277 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 277 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 277 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 277 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 278 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_z16
# 281 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc2d_cn ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 )
# 283 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 284 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 285 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 285 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 285 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 285 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 285 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 285 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 285 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 286 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= element ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_alloc2d_cn
 END INTERFACE
# 292 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_alloc3d
# 297 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_i2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 299 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 300 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 301 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 301 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 301 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 301 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 301 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 301 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 301 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 301 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 301 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 303 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_i2
# 306 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_i4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 308 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 309 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 310 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 310 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 310 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 310 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 310 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 310 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 310 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 310 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 310 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 312 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_i4
# 315 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_i8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 317 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 318 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 319 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 319 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 319 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 319 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 319 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 319 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 319 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 319 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 319 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 321 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_i8
# 324 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_l2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 326 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 327 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 328 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 328 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 328 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 328 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 328 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 328 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 328 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 328 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 328 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 330 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_l2
# 333 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_l4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 335 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 336 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 337 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 337 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 337 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 337 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 337 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 337 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 337 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 337 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 337 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 339 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_l4
# 342 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_l8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 344 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 345 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 346 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 346 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 346 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 346 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 346 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 346 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 346 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 346 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 346 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 348 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_l8
# 351 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_r4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 353 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 354 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 355 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 355 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 355 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 355 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 355 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 355 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 355 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 355 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 355 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 357 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_r4
# 360 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_r8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 362 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 363 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 364 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 364 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 364 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 364 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 364 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 364 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 364 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 364 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 364 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 366 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_r8
# 369 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_z8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 371 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 372 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 373 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 373 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 373 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 373 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 373 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 373 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 373 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 373 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 373 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 375 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_z8
# 378 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_z16 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 380 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 381 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 382 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 382 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 382 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 382 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 382 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 382 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 382 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 382 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 382 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 384 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_z16
# 387 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc3d_cn ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 )
# 389 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 390 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 391 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 391 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 391 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 391 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 391 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 391 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 391 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 391 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 391 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 393 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= element ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_alloc3d_cn
 END INTERFACE
# 399 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_alloc4d
# 404 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_i2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 406 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 407 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 408 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 410 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_i2
# 413 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_i4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 415 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 416 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 417 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 419 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_i4
# 422 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_i8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 424 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 425 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 426 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 428 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_i8
# 431 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_l2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 433 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 434 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 435 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 437 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_l2
# 440 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_l4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 442 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 443 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 444 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 446 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_l4
# 449 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_l8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 451 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 452 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 453 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 455 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_l8
# 458 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_r4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 460 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 461 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 462 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 464 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_r4
# 467 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_r8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 469 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 470 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 471 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 473 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_r8
# 476 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_z8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 478 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 479 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 480 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 482 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_z8
# 485 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_z16 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 487 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 488 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 489 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 491 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_z16
# 494 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc4d_cn ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 )
# 496 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 497 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 498 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 500 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= element ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc4d_cn
 END INTERFACE
# 506 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_alloc5d
# 511 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_i2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 513 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 514 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 515 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 517 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_i2
# 520 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_i4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 522 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 523 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 524 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 526 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_i4
# 529 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_i8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 531 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 532 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 533 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 535 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_i8
# 538 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_l2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 540 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 541 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 542 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 544 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_l2
# 547 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_l4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 549 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 550 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 551 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 553 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_l4
# 556 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_l8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 558 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 559 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 560 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 562 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_l8
# 565 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_r4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 567 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 568 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 569 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 571 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_r4
# 574 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_r8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 576 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 577 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 578 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 580 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_r8
# 583 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_z8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 585 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 586 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 587 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 589 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_z8
# 592 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_z16 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 594 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 595 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 596 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 598 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_z16
# 601 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc5d_cn ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 )
# 603 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 604 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 605 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 607 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= element ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc5d_cn
 END INTERFACE
# 613 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_alloc6d
# 618 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_i2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 621 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 622 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 623 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 625 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_i2
# 628 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_i4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 631 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 632 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 633 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 635 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_i4
# 638 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_i8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 641 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 642 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 643 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 645 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_i8
# 648 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_l2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 651 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 652 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 653 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 655 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_l2
# 658 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_l4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 661 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 662 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 663 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 665 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_l4
# 668 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_l8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 671 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 672 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 673 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 675 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_l8
# 678 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_r4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 681 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 682 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 683 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 685 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_r4
# 688 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_r8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 691 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 692 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 693 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 695 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_r8
# 698 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_z8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 701 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 702 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 703 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 705 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_z8
# 708 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_z16 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 711 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 712 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 713 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 715 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_z16
# 718 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc6d_cn ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 )
# 721 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 722 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 723 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 725 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= element ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc6d_cn
 END INTERFACE
# 731 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_alloc7d
# 736 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_i2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 739 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 740 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 741 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 743 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_i2
# 746 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_i4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 749 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 750 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 751 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 753 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_i4
# 756 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_i8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 759 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 760 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 761 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 763 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_i8
# 766 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_l2 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 769 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 770 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 771 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 773 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_l2
# 776 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_l4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 779 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 780 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 781 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 783 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_l4
# 786 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_l8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 789 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 790 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 791 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 793 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_l8
# 796 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_r4 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 799 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 800 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 801 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 803 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_r4
# 806 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_r8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 809 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 810 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 811 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 813 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_r8
# 816 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_z8 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 819 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 820 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 821 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 823 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_z8
# 826 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_z16 ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 829 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 830 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 831 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 833 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_z16
# 836 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_alloc7d_cn ( descptr , var , count , element , tag , rank , lb1 , ub1 , lb2 , ub2 , lb3 , ub3 , lb4 ,&
   ub4 , lb5 , ub5 , lb6 , ub6 , lb7 , ub7 )
# 839 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(INOUT) :: descptr
# 840 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: count
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: element
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: rank
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb1
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub1
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb2
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub2
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb3
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub3
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb4
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub4
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb5
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub5
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb6
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub6
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: lb7
# 841 "xmp_lib_coarray_alloc.h"
   INTEGER , INTENT(IN) :: ub7
# 843 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= element ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_alloc7d_cn
 END INTERFACE
# 849 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_dealloc0d
# 854 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_i2 ( descptr , var , tag )
# 855 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 855 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 856 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_i2
# 859 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_i4 ( descptr , var , tag )
# 860 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 860 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 861 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_i4
# 864 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_i8 ( descptr , var , tag )
# 865 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 865 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 866 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_i8
# 869 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_l2 ( descptr , var , tag )
# 870 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 870 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 871 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_l2
# 874 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_l4 ( descptr , var , tag )
# 875 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 875 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 876 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_l4
# 879 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_l8 ( descptr , var , tag )
# 880 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 880 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 881 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_l8
# 884 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_r4 ( descptr , var , tag )
# 885 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 885 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 886 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_r4
# 889 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_r8 ( descptr , var , tag )
# 890 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 890 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 891 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_r8
# 894 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_z8 ( descptr , var , tag )
# 895 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 895 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 896 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_z8
# 899 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_z16 ( descptr , var , tag )
# 900 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 900 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 901 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_z16
# 904 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc0d_cn ( descptr , var , tag )
# 905 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 905 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 906 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= * ) , POINTER , INTENT(OUT) :: var
  END SUBROUTINE xmpf_coarray_dealloc0d_cn
 END INTERFACE
# 912 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_dealloc1d
# 917 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_i2 ( descptr , var , tag )
# 918 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 918 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 919 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_i2
# 922 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_i4 ( descptr , var , tag )
# 923 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 923 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 924 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_i4
# 927 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_i8 ( descptr , var , tag )
# 928 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 928 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 929 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_i8
# 932 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_l2 ( descptr , var , tag )
# 933 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 933 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 934 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_l2
# 937 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_l4 ( descptr , var , tag )
# 938 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 938 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 939 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_l4
# 942 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_l8 ( descptr , var , tag )
# 943 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 943 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 944 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_l8
# 947 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_r4 ( descptr , var , tag )
# 948 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 948 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 949 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_r4
# 952 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_r8 ( descptr , var , tag )
# 953 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 953 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 954 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_r8
# 957 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_z8 ( descptr , var , tag )
# 958 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 958 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 959 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_z8
# 962 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_z16 ( descptr , var , tag )
# 963 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 963 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 964 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_z16
# 967 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc1d_cn ( descptr , var , tag )
# 968 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 968 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 969 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= * ) , POINTER , INTENT(OUT) :: var ( : )
  END SUBROUTINE xmpf_coarray_dealloc1d_cn
 END INTERFACE
# 975 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_dealloc2d
# 980 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_i2 ( descptr , var , tag )
# 981 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 981 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 982 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_i2
# 985 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_i4 ( descptr , var , tag )
# 986 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 986 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 987 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_i4
# 990 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_i8 ( descptr , var , tag )
# 991 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 991 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 992 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_i8
# 995 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_l2 ( descptr , var , tag )
# 996 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 996 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 997 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_l2
# 1000 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_l4 ( descptr , var , tag )
# 1001 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1001 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1002 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_l4
# 1005 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_l8 ( descptr , var , tag )
# 1006 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1006 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1007 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_l8
# 1010 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_r4 ( descptr , var , tag )
# 1011 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1011 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1012 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_r4
# 1015 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_r8 ( descptr , var , tag )
# 1016 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1016 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1017 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_r8
# 1020 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_z8 ( descptr , var , tag )
# 1021 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1021 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1022 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_z8
# 1025 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_z16 ( descptr , var , tag )
# 1026 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1026 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1027 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_z16
# 1030 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc2d_cn ( descptr , var , tag )
# 1031 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1031 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1032 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= * ) , POINTER , INTENT(OUT) :: var ( : , : )
  END SUBROUTINE xmpf_coarray_dealloc2d_cn
 END INTERFACE
# 1038 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_dealloc3d
# 1043 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_i2 ( descptr , var , tag )
# 1044 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1044 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1045 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_i2
# 1048 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_i4 ( descptr , var , tag )
# 1049 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1049 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1050 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_i4
# 1053 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_i8 ( descptr , var , tag )
# 1054 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1054 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1055 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_i8
# 1058 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_l2 ( descptr , var , tag )
# 1059 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1059 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1060 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_l2
# 1063 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_l4 ( descptr , var , tag )
# 1064 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1064 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1065 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_l4
# 1068 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_l8 ( descptr , var , tag )
# 1069 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1069 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1070 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_l8
# 1073 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_r4 ( descptr , var , tag )
# 1074 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1074 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1075 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_r4
# 1078 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_r8 ( descptr , var , tag )
# 1079 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1079 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1080 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_r8
# 1083 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_z8 ( descptr , var , tag )
# 1084 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1084 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1085 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_z8
# 1088 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_z16 ( descptr , var , tag )
# 1089 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1089 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1090 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_z16
# 1093 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc3d_cn ( descptr , var , tag )
# 1094 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1094 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1095 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= * ) , POINTER , INTENT(OUT) :: var ( : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc3d_cn
 END INTERFACE
# 1101 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_dealloc4d
# 1106 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_i2 ( descptr , var , tag )
# 1107 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1107 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1108 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_i2
# 1111 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_i4 ( descptr , var , tag )
# 1112 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1112 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1113 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_i4
# 1116 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_i8 ( descptr , var , tag )
# 1117 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1117 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1118 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_i8
# 1121 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_l2 ( descptr , var , tag )
# 1122 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1122 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1123 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_l2
# 1126 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_l4 ( descptr , var , tag )
# 1127 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1127 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1128 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_l4
# 1131 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_l8 ( descptr , var , tag )
# 1132 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1132 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1133 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_l8
# 1136 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_r4 ( descptr , var , tag )
# 1137 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1137 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1138 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_r4
# 1141 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_r8 ( descptr , var , tag )
# 1142 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1142 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1143 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_r8
# 1146 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_z8 ( descptr , var , tag )
# 1147 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1147 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1148 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_z8
# 1151 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_z16 ( descptr , var , tag )
# 1152 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1152 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1153 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_z16
# 1156 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc4d_cn ( descptr , var , tag )
# 1157 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1157 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1158 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= * ) , POINTER , INTENT(OUT) :: var ( : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc4d_cn
 END INTERFACE
# 1164 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_dealloc5d
# 1169 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_i2 ( descptr , var , tag )
# 1170 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1170 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1171 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_i2
# 1174 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_i4 ( descptr , var , tag )
# 1175 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1175 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1176 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_i4
# 1179 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_i8 ( descptr , var , tag )
# 1180 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1180 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1181 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_i8
# 1184 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_l2 ( descptr , var , tag )
# 1185 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1185 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1186 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_l2
# 1189 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_l4 ( descptr , var , tag )
# 1190 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1190 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1191 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_l4
# 1194 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_l8 ( descptr , var , tag )
# 1195 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1195 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1196 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_l8
# 1199 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_r4 ( descptr , var , tag )
# 1200 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1200 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1201 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_r4
# 1204 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_r8 ( descptr , var , tag )
# 1205 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1205 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1206 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_r8
# 1209 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_z8 ( descptr , var , tag )
# 1210 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1210 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1211 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_z8
# 1214 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_z16 ( descptr , var , tag )
# 1215 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1215 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1216 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_z16
# 1219 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc5d_cn ( descptr , var , tag )
# 1220 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1220 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1221 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= * ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc5d_cn
 END INTERFACE
# 1227 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_dealloc6d
# 1232 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_i2 ( descptr , var , tag )
# 1233 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1233 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1234 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_i2
# 1237 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_i4 ( descptr , var , tag )
# 1238 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1238 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1239 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_i4
# 1242 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_i8 ( descptr , var , tag )
# 1243 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1243 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1244 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_i8
# 1247 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_l2 ( descptr , var , tag )
# 1248 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1248 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1249 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_l2
# 1252 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_l4 ( descptr , var , tag )
# 1253 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1253 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1254 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_l4
# 1257 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_l8 ( descptr , var , tag )
# 1258 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1258 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1259 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_l8
# 1262 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_r4 ( descptr , var , tag )
# 1263 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1263 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1264 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_r4
# 1267 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_r8 ( descptr , var , tag )
# 1268 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1268 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1269 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_r8
# 1272 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_z8 ( descptr , var , tag )
# 1273 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1273 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1274 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_z8
# 1277 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_z16 ( descptr , var , tag )
# 1278 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1278 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1279 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_z16
# 1282 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc6d_cn ( descptr , var , tag )
# 1283 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1283 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1284 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= * ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc6d_cn
 END INTERFACE
# 1290 "xmp_lib_coarray_alloc.h"
 INTERFACE xmpf_coarray_dealloc7d
# 1295 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_i2 ( descptr , var , tag )
# 1296 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1296 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1297 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_i2
# 1300 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_i4 ( descptr , var , tag )
# 1301 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1301 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1302 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_i4
# 1305 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_i8 ( descptr , var , tag )
# 1306 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1306 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1307 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_i8
# 1310 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_l2 ( descptr , var , tag )
# 1311 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1311 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1312 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 2 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_l2
# 1315 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_l4 ( descptr , var , tag )
# 1316 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1316 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1317 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_l4
# 1320 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_l8 ( descptr , var , tag )
# 1321 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1321 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1322 "xmp_lib_coarray_alloc.h"
   LOGICAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_l8
# 1325 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_r4 ( descptr , var , tag )
# 1326 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1326 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1327 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_r4
# 1330 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_r8 ( descptr , var , tag )
# 1331 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1331 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1332 "xmp_lib_coarray_alloc.h"
   REAL ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_r8
# 1335 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_z8 ( descptr , var , tag )
# 1336 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1336 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1337 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 4 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_z8
# 1340 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_z16 ( descptr , var , tag )
# 1341 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1341 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1342 "xmp_lib_coarray_alloc.h"
   COMPLEX ( KIND= 8 ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_z16
# 1345 "xmp_lib_coarray_alloc.h"
  SUBROUTINE xmpf_coarray_dealloc7d_cn ( descptr , var , tag )
# 1346 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: descptr
# 1346 "xmp_lib_coarray_alloc.h"
   INTEGER ( KIND= 8 ) , INTENT(IN) :: tag
# 1347 "xmp_lib_coarray_alloc.h"
   CHARACTER ( LEN= * ) , POINTER , INTENT(OUT) :: var ( : , : , : , : , : , : , : )
  END SUBROUTINE xmpf_coarray_dealloc7d_cn
 END INTERFACE
 COMMON / XMP_COMMON / xmp_size_array

 CALL xmpf_coarray_proc_init ( xmpf_resource_tag )
# 6 "allo1.f90"
 me = this_image ( )
# 7 "allo1.f90"
 nerr = 0
# 9 "allo1.f90"
 IF ( associated ( a ) ) THEN
# 10 "allo1.f90"
  nerr = nerr + 1
  WRITE ( unit = * , fmt = * )"1. allocated(a) must be false but:" , associated ( a )
 END IF
# 14 "allo1.f90"
 CALL xmpf_coarray_alloc1d ( xmpf_descptr_a , a , 10 , 4 , xmpf_resource_tag , 1 , 1 , 10 )
# 14 "allo1.f90"
 CALL xmpf_coarray_set_coshape ( xmpf_descptr_a , 1 , 1 )
# 15 "allo1.f90"
 ALLOCATE ( b ( 10 ) )
# 17 "allo1.f90"
 IF ( ( .NOT. associated ( a ) ) ) THEN
# 18 "allo1.f90"
  nerr = nerr + 1
  WRITE ( unit = * , fmt = * )"2. allocated(a) must be true but:" , associated ( a )
 END IF
# 22 "allo1.f90"
 CALL xmpf_coarray_dealloc1d ( xmpf_descptr_a , a , xmpf_resource_tag )
# 23 "allo1.f90"
 DEALLOCATE ( b )
# 25 "allo1.f90"
 IF ( associated ( a ) ) THEN
# 26 "allo1.f90"
  nerr = nerr + 1
  WRITE ( unit = * , fmt = * )"3. allocated(a) must be false but:" , associated ( a )
 END IF
# 30 "allo1.f90"
 CALL xmpf_coarray_alloc1d ( xmpf_descptr_a , a , 10000 , 4 , xmpf_resource_tag , 1 , 1 , 10000 )
# 30 "allo1.f90"
 CALL xmpf_coarray_set_coshape ( xmpf_descptr_a , 1 , 1 )
# 31 "allo1.f90"
 ALLOCATE ( b ( 10000 ) )
# 33 "allo1.f90"
 IF ( ( .NOT. associated ( a ) ) ) THEN
# 34 "allo1.f90"
  nerr = nerr + 1
  WRITE ( unit = * , fmt = * )"4. allocated(a) must be true but:" , associated ( a )
 END IF
# 38 "allo1.f90"
 IF ( nerr == 0 ) THEN
# 39 "allo1.f90"
  PRINT '("[",i0,"] OK")' , me
 ELSE
# 41 "allo1.f90"
  PRINT '("[",i0,"] number of NGs: ",i0)' , me , nerr
 END IF
 CALL xmpf_coarray_proc_finalize ( xmpf_resource_tag )
99999 &
 CONTINUE
END SUBROUTINE xmpf_main

