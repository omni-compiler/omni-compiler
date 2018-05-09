package xcodeml.c.type;

import java.util.HashMap;
import java.util.Map;

public enum XcBaseTypeEnum
{
    VOID                    ("void",                    "void",                     null,   false),
    CHAR                    ("char",                    "char",                     null,   false),
    SHORT                   ("short",                   "short",                    null,   false),
    INT                     ("int",                     "int",                      null,   false),
    LONG                    ("long",                    "long",                     "l",    false),
    LONGLONG                ("long_long",               "long long",                "ll",   false),
    UCHAR                   ("unsigned_char",           "unsigned char",            null,   true),
    USHORT                  ("unsigned_short",          "unsigned short",           null,   true),
    UINT                    ("unsigned",                "unsigned int",             "u",    true),
    ULONG                   ("unsigned_long",           "unsigned long",            "ul",   true),
    ULONGLONG               ("unsigned_long_long",      "unsigned long long",       "ull",  true),
    FLOAT                   ("float",                   "float",                    null,    false),
    DOUBLE                  ("double",                  "double",                   null,   false),
    LONGDOUBLE              ("long_double",             "long double",              null,    false),
    BOOL                    ("bool",                    "_Bool",                    null,   true), /* _Bool is treated as unsigned */
    WCHAR                   ("wchar_t",                 "wchar_t",                  null,   false),
    FLOAT_COMPLEX           ("float_complex",           "float _Complex",           null,   false),
    DOUBLE_COMPLEX          ("double_complex",          "double _Complex",          null,   false),
    LONGDOUBLE_COMPLEX      ("long_double_complex",     "long double _Complex",     null,   false),
    FLOAT_IMAG              ("float_imaginary",         "float _Complex",           null,   false),
    DOUBLE_IMAG             ("double_imaginary",        "double _Complex",          null,   false),
    LONGDOUBLE_IMAG         ("long_double_imaginary",   "long double _Complex",     null,   false),
    GCC_BUILTIN_VA_LIST     ("__builtin_va_list",       "__builtin_va_list",        null,   false),
    ;
    
    private final String _xcode, _ckeyword, _literalSuffix;
    
    private final boolean _isUnsigned;
    
    private static final Map<XcBaseTypeEnum, XcBaseType> s_typeMap = new HashMap<XcBaseTypeEnum, XcBaseType>();
    
    private static final Map<XcBaseTypeEnum, XcBaseType> s_constTypeMap = new HashMap<XcBaseTypeEnum, XcBaseType>();
    
    private static final Map<String, XcBaseTypeEnum> s_xcodeMap = new HashMap<String, XcBaseTypeEnum>();
    
    static
    {
        s_typeMap.put(VOID,                 new XcVoidType());
        s_typeMap.put(CHAR,                 new XcIntegerType.Char());
        s_typeMap.put(SHORT,                new XcIntegerType.Short());
        s_typeMap.put(INT,                  new XcIntegerType.Int());
        s_typeMap.put(LONG,                 new XcIntegerType.Long());
        s_typeMap.put(LONGLONG,             new XcIntegerType.LongLong());
        s_typeMap.put(UCHAR,                new XcIntegerType.UChar());
        s_typeMap.put(USHORT,               new XcIntegerType.UShort());
        s_typeMap.put(UINT,                 new XcIntegerType.UInt());
        s_typeMap.put(ULONG,                new XcIntegerType.ULong());
        s_typeMap.put(ULONGLONG,            new XcIntegerType.ULongLong());
        s_typeMap.put(BOOL,                 new XcIntegerType.Bool());
        s_typeMap.put(WCHAR,                new XcIntegerType.Wchar());
        s_typeMap.put(FLOAT,                new XcDecimalType.Float());
        s_typeMap.put(DOUBLE,               new XcDecimalType.Double());
        s_typeMap.put(LONGDOUBLE,           new XcDecimalType.LongDouble());
        s_typeMap.put(FLOAT_COMPLEX,        new XcComplexType.FloatComplex());
        s_typeMap.put(DOUBLE_COMPLEX,       new XcComplexType.DoubleComplex());
        s_typeMap.put(LONGDOUBLE_COMPLEX,   new XcComplexType.LongDoubleComplex());
        s_typeMap.put(FLOAT_IMAG,           new XcImagType.FloatImag());
        s_typeMap.put(DOUBLE_IMAG,          new XcImagType.DoubleImag());
        s_typeMap.put(LONGDOUBLE_IMAG,      new XcImagType.LongDoubleImag());
        s_typeMap.put(GCC_BUILTIN_VA_LIST,  new XcGccBuiltinVaListType());

        s_constTypeMap.put(VOID,                 _const(new XcVoidType()));
        s_constTypeMap.put(CHAR,                 _const(new XcIntegerType.Char()));
        s_constTypeMap.put(SHORT,                _const(new XcIntegerType.Short()));
        s_constTypeMap.put(INT,                  _const(new XcIntegerType.Int()));
        s_constTypeMap.put(LONG,                 _const(new XcIntegerType.Long()));
        s_constTypeMap.put(LONGLONG,             _const(new XcIntegerType.LongLong()));
        s_constTypeMap.put(UCHAR,                _const(new XcIntegerType.UChar()));
        s_constTypeMap.put(USHORT,               _const(new XcIntegerType.UShort()));
        s_constTypeMap.put(UINT,                 _const(new XcIntegerType.UInt()));
        s_constTypeMap.put(ULONG,                _const(new XcIntegerType.ULong()));
        s_constTypeMap.put(ULONGLONG,            _const(new XcIntegerType.ULongLong()));
        s_constTypeMap.put(BOOL,                 _const(new XcIntegerType.Bool()));
        s_constTypeMap.put(WCHAR,                _const(new XcIntegerType.Wchar()));
        s_constTypeMap.put(FLOAT,                _const(new XcDecimalType.Float()));
        s_constTypeMap.put(DOUBLE,               _const(new XcDecimalType.Double()));
        s_constTypeMap.put(LONGDOUBLE,           _const(new XcDecimalType.LongDouble()));
        s_constTypeMap.put(FLOAT_COMPLEX,        _const(new XcComplexType.FloatComplex()));
        s_constTypeMap.put(DOUBLE_COMPLEX,       _const(new XcComplexType.DoubleComplex()));
        s_constTypeMap.put(LONGDOUBLE_COMPLEX,   _const(new XcComplexType.LongDoubleComplex()));
        s_constTypeMap.put(FLOAT_IMAG,           _const(new XcImagType.FloatImag()));
        s_constTypeMap.put(DOUBLE_IMAG,          _const(new XcImagType.DoubleImag()));
        s_constTypeMap.put(LONGDOUBLE_IMAG,      _const(new XcImagType.LongDoubleImag()));
        s_constTypeMap.put(LONGDOUBLE_IMAG,      _const(new XcGccBuiltinVaListType()));
        
        s_xcodeMap.put(VOID._xcode,                 VOID);
        s_xcodeMap.put(CHAR._xcode,                 CHAR);
        s_xcodeMap.put(SHORT._xcode,                SHORT);
        s_xcodeMap.put(INT._xcode,                  INT);
        s_xcodeMap.put(LONG._xcode,                 LONG);
        s_xcodeMap.put(LONGLONG._xcode,             LONGLONG);
        s_xcodeMap.put(UCHAR._xcode,                UCHAR);
        s_xcodeMap.put(USHORT._xcode,               USHORT);
        s_xcodeMap.put(UINT._xcode,                 UINT);
        s_xcodeMap.put(ULONG._xcode,                ULONG);
        s_xcodeMap.put(ULONGLONG._xcode,            ULONGLONG);
        s_xcodeMap.put(BOOL._xcode,                 BOOL);
        s_xcodeMap.put(WCHAR._xcode,                WCHAR);
        s_xcodeMap.put(FLOAT._xcode,                FLOAT);
        s_xcodeMap.put(DOUBLE._xcode,               DOUBLE);
        s_xcodeMap.put(LONGDOUBLE._xcode,           LONGDOUBLE);
        s_xcodeMap.put(FLOAT_COMPLEX._xcode,        FLOAT_COMPLEX);
        s_xcodeMap.put(DOUBLE_COMPLEX._xcode,       DOUBLE_COMPLEX);
        s_xcodeMap.put(LONGDOUBLE_COMPLEX._xcode,   LONGDOUBLE_COMPLEX);
        s_xcodeMap.put(FLOAT_IMAG._xcode,           FLOAT_IMAG);
        s_xcodeMap.put(DOUBLE_IMAG._xcode,          DOUBLE_IMAG);
        s_xcodeMap.put(LONGDOUBLE_IMAG._xcode,      LONGDOUBLE_IMAG);
        s_xcodeMap.put(GCC_BUILTIN_VA_LIST._xcode,  GCC_BUILTIN_VA_LIST);
    }

    private XcBaseTypeEnum(String xcode, String ckeyword, String literalSuffix, boolean isUnigned)
    {
        _xcode = xcode;
        _ckeyword = ckeyword;
        _literalSuffix = literalSuffix;
        _isUnsigned = isUnigned;
    }
    
    private static XcBaseType _const(XcBaseType bt)
    {
        bt.setIsConst(true);
        return bt;
    }
    
    public final String getXcodeKeyword()
    {
        return _xcode;
    }
    
    public final String getCCode()
    {
        return _ckeyword;
    }
    
    public final String getLiteralSuffix()
    {
        return _literalSuffix;
    }
    
    public final boolean isUnsigned()
    {
        return _isUnsigned;
    }
    
    public final XcBaseType getSingletonType()
    {
        return s_typeMap.get(this);
    }
    
    public final XcBaseType getSingletonConstType()
    {
        return s_constTypeMap.get(this);
    }
    
    public static final XcBaseTypeEnum getByXcode(String xcode)
    {
        return s_xcodeMap.get(xcode);
    }
    public final XcBaseType createType()
    {
        switch(this)
        {
        case VOID:
            return new XcVoidType();
        case CHAR:
            return new XcIntegerType.Char();
        case SHORT:
            return new XcIntegerType.Short();
        case INT:
            return new XcIntegerType.Int();
        case LONG:
            return new XcIntegerType.Long();
        case LONGLONG:
            return new XcIntegerType.LongLong();
        case UCHAR:
            return new XcIntegerType.UChar();
        case USHORT:
            return new XcIntegerType.UShort();
        case UINT:
            return new XcIntegerType.UInt();
        case ULONG:
            return new XcIntegerType.ULong();
        case ULONGLONG:
            return new XcIntegerType.ULongLong();
        case BOOL:
            return new XcIntegerType.Bool();
        case WCHAR:
            return new XcIntegerType.Wchar();
        case FLOAT:
            return new XcDecimalType.Float();
        case DOUBLE:
            return new XcDecimalType.Double();
        case LONGDOUBLE:
            return new XcDecimalType.LongDouble();
        case FLOAT_COMPLEX:
            return new XcComplexType.FloatComplex();
        case DOUBLE_COMPLEX:
            return new XcComplexType.DoubleComplex();
        case LONGDOUBLE_COMPLEX:
            return new XcComplexType.LongDoubleComplex();
        case FLOAT_IMAG:
            return new XcImagType.FloatImag();
        case DOUBLE_IMAG:
            return new XcImagType.DoubleImag();
        case LONGDOUBLE_IMAG:
            return new XcImagType.LongDoubleImag();
        case GCC_BUILTIN_VA_LIST:
            return new XcGccBuiltinVaListType();
        default:
            /* not reachable */
            throw new IllegalArgumentException(this.toString());
        }
    }

    public static final boolean isBuiltInType(String xcode)
    {
        return s_xcodeMap.containsKey(xcode);
    }
    
    public static final XcBaseType createTypeByXcode(String xcode)
    {
        XcBaseTypeEnum btEnum = s_xcodeMap.get(xcode);
        
        if(btEnum == null)
            return null;
        
        return btEnum.createType();
    }
}
