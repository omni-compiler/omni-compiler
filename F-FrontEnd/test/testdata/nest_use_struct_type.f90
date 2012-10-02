      ! 2段以上の参照結合を挟んだ構造型の参照
      PROGRAM MAIN
        USE use_struct_type
        TYPE(t) :: a
        a%n = 1
      end PROGRAM MAIN
