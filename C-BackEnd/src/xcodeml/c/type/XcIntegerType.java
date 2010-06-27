/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

/**
 * type of 'char', 'short', 'int', 'long', 'long long'
 */
public abstract class XcIntegerType extends XcNumType
{
    public XcIntegerType(XcBaseTypeEnum basicTypeEnum, String typeId)
    {
        super(basicTypeEnum, typeId);
    }
    
    public static final class Char extends XcIntegerType
    {
        public Char()
        {
            this(null);
        }
        
        public Char(String typeId)
        {
            super(XcBaseTypeEnum.CHAR, typeId);
        }
    }
    
    public static final class Short extends XcIntegerType
    {
        public Short()
        {
            this(null);
        }
        
        public Short(String typeId)
        {
            super(XcBaseTypeEnum.SHORT, typeId);
        }
    }
    
    public static final class Int extends XcIntegerType
    {
        public Int()
        {
            this(null);
        }
        
        public Int(String typeId)
        {
            super(XcBaseTypeEnum.INT, typeId);
        }
    }

    public static final class Long extends XcIntegerType
    {
        public Long()
        {
            this(null);
        }
        
        public Long(String typeId)
        {
            super(XcBaseTypeEnum.LONG, typeId);
        }
    }

    public static final class LongLong extends XcIntegerType
    {
        public LongLong()
        {
            this(null);
        }
        
        public LongLong(String typeId)
        {
            super(XcBaseTypeEnum.LONGLONG, typeId);
        }
    }

    public static final class Bool extends XcIntegerType
    {
        public Bool()
        {
            this(null);
        }
        
        public Bool(String typeId)
        {
            super(XcBaseTypeEnum.BOOL, typeId);
        }
    }

    public static final class Wchar extends XcIntegerType
    {
        public Wchar()
        {
            this(null);
        }
        
        public Wchar(String typeId)
        {
            super(XcBaseTypeEnum.WCHAR, typeId);
        }
    }

    public static final class UChar extends XcIntegerType
    {
        public UChar()
        {
            this(null);
        }
        
        public UChar(String typeId)
        {
            super(XcBaseTypeEnum.UCHAR, typeId);
        }
    }

    public static final class UShort extends XcIntegerType
    {
        public UShort()
        {
            this(null);
        }
        
        public UShort(String typeId)
        {
            super(XcBaseTypeEnum.USHORT, typeId);
        }
    }
    
    public static final class UInt extends XcIntegerType
    {
        public UInt()
        {
            this(null);
        }
        
        public UInt(String typeId)
        {
            super(XcBaseTypeEnum.UINT, typeId);
        }
    }

    public static final class ULong extends XcIntegerType
    {
        public ULong()
        {
            this(null);
        }
        
        public ULong(String typeId)
        {
            super(XcBaseTypeEnum.ULONG, typeId);
        }
    }

    public static final class ULongLong extends XcIntegerType
    {
        public ULongLong()
        {
            this(null);
        }
        
        public ULongLong(String typeId)
        {
            super(XcBaseTypeEnum.ULONGLONG, typeId);
        }
    }
}
