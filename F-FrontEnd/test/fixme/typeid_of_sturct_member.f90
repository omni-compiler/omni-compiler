Module m

TYPE a
    INTEGER :: b
END TYPE

Type btree
    Type (btree), Pointer :: next
End type

Contains

Function f(p)
    Type (btree) :: p
    integer f
    f = 1
End Function

End Module
