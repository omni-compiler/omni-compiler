       PROGRAM main
         TYPE t
           INTEGER :: i = 0
           PROCEDURE(REAL), NOPASS, POINTER :: v         ! 返り型 REAL を持つ関数への手続きポインタ

           PROCEDURE(f), PASS(a), POINTER :: u => null() ! 関数 f への手続きポインタ

          CONTAINS
           PROCEDURE, PASS :: w => f                     ! こちらは型束縛手続き
         END TYPE t

         INTERFACE
           FUNCTION f(a)
             IMPORT t         ! 派生型 t を使うために IMPORT 文が必要
             INTEGER :: f
             CLASS(t) :: a    ! 型束縛手続き同様に、PASS で指定された引数は
                              ! CLASS にする必要がある
           END FUNCTION f
         END INTERFACE

         TYPE(t) :: v
         INTEGER :: i

         v%u => g
         PRINT *, v%u() ! -> 1
         PRINT *, v%u() ! -> 2
         PRINT *, v%u() ! -> 3
         
       CONTAINS
         FUNCTION g(a)
           INTEGER :: g
           CLASS(t) :: a
           a%i = a%i + 1
           g = a%i
         END FUNCTION g
       END PROGRAM main

       FUNCTION f(a)
         INTEGER :: f
         CLASS(*) :: a
       END FUNCTION f
