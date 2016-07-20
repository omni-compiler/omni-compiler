program Hello_World
  implicit none
  integer :: i  ! Local variable
  character(len=20) :: name(1)[*] ! scalar coarray
  ! 注意: "name[<index>]" はリモートのイメージ上の変数への
  ! アクセスであるのに対し "name" はローカル変数である

  ! イメージ1上のユーザーから名前の入力を受ける
  if (this_image() == 1) then
    write(*,'(a)',advance='no') 'Enter your name: '
    read(*,'(a)') name

    ! 他のイメージに名前の内容を分配する。
    do i = 2, num_images()
      name[i] = name
    end do
  end if

  sync all ! 確実に同期をとるために[[バリア]]を設ける

  ! すべてノードで名前を表示する
  write(*,'(3a,i0)') 'Hello ',trim(name(1)),' from image ', this_image()
end program Hello_world
