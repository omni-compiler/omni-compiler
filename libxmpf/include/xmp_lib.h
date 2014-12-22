      type xmp_desc
        sequence
        integer*8 :: desc
      end type xmp_desc

      interface xmpf_coarray_get_array

!!      integer, 2/4/8 bytes
        function xmpf_coarray_get_array_i2(serno, addr0, coindex, rank, &
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        integer(2) :: val
        integer(2), intent(in) :: addr0
        integer(2), optional, intent(in) ::                             &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

        function xmpf_coarray_get_array_i4(serno, addr0, coindex, rank, &
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        integer(4) :: val
        integer(4), intent(in) :: addr0
        integer(4), optional, intent(in) ::                             &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

        function xmpf_coarray_get_array_i8(serno, addr0, coindex, rank, &
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        integer(8) :: val
        integer(8), intent(in) :: addr0
        integer(8), optional, intent(in) ::                             &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

!!      logical, 2/4/8 bytes
        function xmpf_coarray_get_array_l2(serno, addr0, coindex, rank, &
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        logical(2) :: val
        logical(2), intent(in) :: addr0
        logical(2), optional, intent(in) ::                             &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

        function xmpf_coarray_get_array_l4(serno, addr0, coindex, rank, &
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        logical(4) :: val
        logical(4), intent(in) :: addr0
        logical(4), optional, intent(in) ::                             &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

        function xmpf_coarray_get_array_l8(serno, addr0, coindex, rank, &
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        logical(8) :: val
        logical(8), intent(in) :: addr0
        logical(8), optional, intent(in) ::                             &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

!!      real, 4/8 bytes
!!        real(kind=16) is not supported in XMP/F
        function xmpf_coarray_get_array_r4(serno, addr0, coindex, rank, &
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        real(4) :: val
        real(4), intent(in) :: addr0
        real(4), optional, intent(in) ::                                &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

        function xmpf_coarray_get_array_r8(serno, addr0, coindex, rank, &
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        real(8) :: val
        real(8), intent(in) :: addr0
        real(8), optional, intent(in) ::                                &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

!!      complex, 8/16/32 bytes
!!        complex(kind=16) (32bytes) is not supported in XMP/F
        function xmpf_coarray_get_array_z8(serno, addr0, coindex, rank, &
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        complex(4) :: val
        complex(4), intent(in) :: addr0
        complex(4), optional, intent(in) ::                             &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

        function xmpf_coarray_get_array_z16(serno, addr0, coindex, rank,&
     &     addr1,  cnt1,  addr2,  cnt2,  addr3,  cnt3,  addr4,  cnt4,   &
     &     addr5,  cnt5,  addr6,  cnt6,  addr7,  cnt7,  addr8,  cnt8,   &
     &     addr9,  cnt9,  addr10, cnt10, addr11, cnt11, addr12, cnt12,  &
     &     addr13, cnt13, addr14, cnt14, addr15, cnt15, addr16, cnt16,  &
     &     addr17, cnt17, addr18, cnt18, addr19, cnt19, addr20, cnt20,  &
     &     addr21, cnt21, addr22, cnt22, addr23, cnt23, addr24, cnt24,  &
     &     addr25, cnt25, addr26, cnt26, addr27, cnt27, addr28, cnt28,  &
     &     addr29, cnt29, addr30, cnt30, addr31, cnt31) result(val)
        complex(8) :: val
        complex(8), intent(in) :: addr0
        complex(8), optional, intent(in) ::                             &
     &       addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8,    &
     &       addr9, addr10,addr11,addr12,addr13,addr14,addr15,addr16,   &
     &       addr17,addr18,addr19,addr20,addr21,addr22,addr23,addr24,   &
     &       addr25,addr26,addr27,addr28,addr29,addr30,addr31
        integer, intent(in) :: serno, coindex, rank
        integer, optional, intent(in) ::                                &
     &       cnt1,  cnt2,  cnt3,  cnt4,  cnt5,  cnt6,  cnt7,  cnt8,     &
     &       cnt9,  cnt10, cnt11, cnt12, cnt13, cnt14, cnt15, cnt16,    &
     &       cnt17, cnt18, cnt19, cnt20, cnt21, cnt22, cnt23, cnt24,    &
     &       cnt25, cnt26, cnt27, cnt28, cnt29, cnt30, cnt31
        end function

      end interface
