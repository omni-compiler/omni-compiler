FUNCTION foo ( n0 , a2 , m0 )
# 5 "dummyarg-1.f90"
 INTEGER :: n0
 EXTERNAL xmpf_coarray_proc_init
 INTEGER ( KIND= 8 ) :: xmpf_descptr_n0
# 1 "dummyarg-1.f90"
 REAL :: foo
# 29 "xmp_lib.h"
 INTEGER :: xmpf_coarray_image
 EXTERNAL xmpf_coarray_image
# 17 "xmp_lib.h"
 INTEGER :: this_image
 EXTERNAL this_image
 INTEGER :: k
 INTEGER :: xmp_size_array ( 0 : 15 , 0 : 6 )
# 4 "dummyarg-1.f90"
 INTEGER :: m0
 EXTERNAL xmpf_coarray_descptr
 INTEGER ( KIND= 8 ) :: xmpf_descptr_a2
# 17 "xmp_lib.h"
 INTEGER :: num_images
 EXTERNAL num_images
 INTEGER ( KIND= 8 ) :: xmpf_coarray_proc_tag
# 6 "dummyarg-1.f90"
 REAL :: a2 ( 1 : 3 , 1 : m0 )
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
 COMMON / XMP_COMMON / xmp_size_array

 XMP_dummy0 = xmpf_foo ( n0 , a2 , m0 )
 CALL xmpf_coarray_proc_init ( xmpf_coarray_proc_tag )
 CALL xmpf_coarray_proc_finalize ( xmpf_coarray_proc_tag )
 CALL xmpf_coarray_descptr ( xmpf_descptr_n0 , n0 )
 CALL xmpf_coarray_descptr ( xmpf_descptr_a2 , a2 )
99999 &
 CONTINUE

CONTAINS
 FUNCTION xmpf_foo ( n0 , a2 , m0 )
  INTEGER :: n0
  INTEGER :: xmpf_coarray_get_image_index
  EXTERNAL xmpf_coarray_get_image_index
  EXTERNAL xmpf_coarray_descptr
  EXTERNAL xmpf_coarray_proc_init
  INTEGER ( KIND= 8 ) :: xmpf_descptr_a2
  INTEGER ( KIND= 8 ) :: xmpf_coarray_proc_tag
  INTEGER ( KIND= 8 ) :: xmpf_descptr_n0
  REAL :: a2 ( 1 : 3 , 1 : m0 )
  EXTERNAL xmpf_coarray_proc_finalize
  INTEGER :: m0

# 8 "dummyarg-1.f90"
  k = xmpf_coarray_get0d ( xmpf_descptr_n0 , n0 , 4 , xmpf_coarray_get_image_index ( xmpf_descptr_n0 , 1 , m0 ) , 0 )
# 9 "dummyarg-1.f90"
  foo = k * xmpf_coarray_get0d ( xmpf_descptr_a2 , a2 ( 2 , 3 ) , 4 , xmpf_coarray_get_image_index ( xmpf_descptr_a2 , 1 , 1 ) , 0&
   )
  GOTO 99999
  CALL xmpf_coarray_proc_init ( xmpf_coarray_proc_tag )
  CALL xmpf_coarray_proc_finalize ( xmpf_coarray_proc_tag )
  CALL xmpf_coarray_descptr ( xmpf_descptr_n0 , n0 )
  CALL xmpf_coarray_descptr ( xmpf_descptr_a2 , a2 )
 99999 &
  CONTINUE
 END FUNCTION xmpf_foo

END FUNCTION foo

SUBROUTINE xmpf_main ( )
 INTEGER ( KIND= 8 ) :: xmpf_descptr_n
 EXTERNAL xmpf_traverse_initcoarray_1xmpf_main
# 17 "dummyarg-1.f90"
 INTEGER :: n
 REAL :: foo
 EXTERNAL xmpf_traverse_coarraysize_1xmpf_main
# 29 "xmp_lib.h"
 INTEGER :: xmpf_coarray_image
 EXTERNAL xmpf_coarray_image
# 17 "xmp_lib.h"
 INTEGER :: this_image
 EXTERNAL this_image
 INTEGER :: nerr
 INTEGER :: i
 INTEGER :: xmp_size_array ( 0 : 15 , 0 : 6 )
# 17 "xmp_lib.h"
 INTEGER :: num_images
 EXTERNAL num_images
 INTEGER ( KIND= 8 ) :: xmpf_descptr_a1
 EXTERNAL xmpf_coarray_share_pool
 POINTER ( xmpf_crayptr_n , n )
# 18 "dummyarg-1.f90"
 REAL :: a1 ( 1 : 300 )
 INTEGER :: me
 EXTERNAL xmpf_coarray_set_coshape
 REAL :: ans
 EXTERNAL xmpf_coarray_count_size
# 4 "xmp_lib.h"
 TYPE :: xmp_desc
  SEQUENCE
  INTEGER ( KIND= 8 ) :: desc
 END TYPE xmp_desc
 POINTER ( xmpf_crayptr_a1 , a1 )
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
 COMMON / xmpf_descptr_xmpf_main / xmpf_descptr_n , xmpf_descptr_a1
 COMMON / xmpf_crayptr_xmpf_main / xmpf_crayptr_n , xmpf_crayptr_a1
 COMMON / XMP_COMMON / xmp_size_array

# 20 "dummyarg-1.f90"
 me = this_image ( )
 DO i = 1 , 300 , 1
# 27 "dummyarg-1.f90"
  a1 ( i ) = float ( me + 10 * i )
 END DO
# 29 "dummyarg-1.f90"
 n = ( - me )
# 30 "dummyarg-1.f90"
 CALL xmpf_sync_all ( )
# 33 "dummyarg-1.f90"
 ans = 0.0
# 34 "dummyarg-1.f90"
 IF ( me == 1 ) THEN
# 35 "dummyarg-1.f90"
  ans = foo ( n , a1 , 3 )
 END IF
# 41 "dummyarg-1.f90"
 CALL xmpf_sync_all ( )
# 44 "dummyarg-1.f90"
 nerr = 0
# 45 "dummyarg-1.f90"
 IF ( me == 1 ) THEN
# 46 "dummyarg-1.f90"
  IF ( ( - 243.0001 ) < ans .AND. ans < ( - 242.9999 ) ) THEN
# 47 "dummyarg-1.f90"
   CONTINUE
  ELSE
# 49 "dummyarg-1.f90"
   nerr = nerr + 1
  END IF
 ELSE
# 52 "dummyarg-1.f90"
  IF ( ( - 0.0001 ) < ans .AND. ans < 0.0001 ) THEN
# 53 "dummyarg-1.f90"
   CONTINUE
  ELSE
# 55 "dummyarg-1.f90"
   nerr = nerr + 1
  END IF
 END IF
# 59 "dummyarg-1.f90"
 IF ( nerr == 0 ) THEN
# 60 "dummyarg-1.f90"
  PRINT '("result[",i0,"] OK")' , me
 ELSE
# 62 "dummyarg-1.f90"
  PRINT '("result[",i0,"] number of NGs: ",i0)' , me , nerr
 END IF
99999 &
 CONTINUE
END SUBROUTINE xmpf_main

SUBROUTINE xmpf_traverse_coarraysize_1xmpf_main ( )

 CALL xmpf_coarray_count_size ( 1 , 4 )
 CALL xmpf_coarray_count_size ( 300 , 4 )
END SUBROUTINE xmpf_traverse_coarraysize_1xmpf_main

SUBROUTINE xmpf_traverse_initcoarray_1xmpf_main ( )
 INTEGER ( KIND= 8 ) :: xmpf_descptr_n
 EXTERNAL xmpf_coarray_share_pool
 INTEGER ( KIND= 8 ) :: xmpf_descptr_a1
 INTEGER ( KIND= 8 ) :: xmpf_crayptr_n
 INTEGER ( KIND= 8 ) :: xmpf_crayptr_a1
 EXTERNAL xmpf_coarray_set_coshape
 COMMON / xmpf_descptr_xmpf_main / xmpf_descptr_n
 COMMON / xmpf_crayptr_xmpf_main / xmpf_crayptr_n
 COMMON / xmpf_descptr_xmpf_main / xmpf_descptr_a1
 COMMON / xmpf_crayptr_xmpf_main / xmpf_crayptr_a1

 CALL xmpf_coarray_share_pool ( xmpf_descptr_n , xmpf_crayptr_n , 1 , 4 ,"n" , 1 )
 CALL xmpf_coarray_set_coshape ( xmpf_descptr_n , 1 , 1 )
 CALL xmpf_coarray_share_pool ( xmpf_descptr_a1 , xmpf_crayptr_a1 , 300 , 4 ,"a1" , 2 )
 CALL xmpf_coarray_set_coshape ( xmpf_descptr_a1 , 1 , 1 )
END SUBROUTINE xmpf_traverse_initcoarray_1xmpf_main

