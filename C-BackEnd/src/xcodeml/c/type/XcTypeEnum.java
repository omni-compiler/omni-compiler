package xcodeml.c.type;

public enum XcTypeEnum
{
    BASICTYPE   ("basic type"),
    BASETYPE    ("base type"),
    STRUCT      ("struct type"),
    UNION       ("union type"),
    ENUM        ("enum type"),
    POINTER     ("pointer type"),
    ARRAY       ("array type"),
    FUNC        ("function type"),
    BUILTIN     ("builtin type"),
    COARRAY     ("coarray type"),
    ;
    
    private String _desc;
    
    private XcTypeEnum(String desc)
    {
        _desc = desc;
    }
    
    public final String getDescription()
    {
        return _desc;
    }
}
