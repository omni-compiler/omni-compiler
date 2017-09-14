module mod1
  use issue268_mod, ONLY: sub1, fct1
contains

  subroutine sub2(sub1, fct1)
    external sub1, fct1
  end subroutine sub2

  subroutine sub3()
    call sub2(sub1, fct1)
  end subroutine sub3
end module mod1
