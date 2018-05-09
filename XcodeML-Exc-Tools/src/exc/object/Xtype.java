package exc.object;

import exc.block.Block;
import exc.util.MachineDepConst;

import xcodeml.util.XmOption;
import static xcodeml.util.XmLog.fatal;

public class Xtype
{
    // type kind
    public final static int UNDEF           = 0;
    public final static int BASIC           = 1;
    public final static int ENUM            = 2;
    public final static int STRUCT          = 3;
    public final static int UNION           = 4;
    public final static int FUNCTION        = 5;
    public final static int ARRAY           = 6;
    public final static int POINTER         = 7;
    public final static int F_ARRAY         = 8;
    public final static int XMP_CO_ARRAY    = 9;
    public final static int F_COARRAY       = 10;        // ID=060

    final static String kind_names[] = {
        "UNDEF",
        "BASIC",
        "ENUM",
        "STRUCT",
        "UNION",
        "FUNCTION",
        "ARRAY",
        "POINTER",
        "F_ARRAY",
        "XMP_COARRAY",
        "F_COARRAY",        // ID=060
    };

    //
    // Qualifiers for C
    //
    public static final long TQ_CONST            = 1L << 0;   // const
    public static final long TQ_VOLATILE         = 1L << 1;   // volatile
    public static final long TQ_RESTRICT         = 1L << 2;   // restrict
    public static final long TQ_INLINE           = 1L << 3;   // inline
    public static final long TQ_ARRAY_STATIC     = 1L << 4;   // static at array specifier
    public static final long TQ_FUNC_STATIC      = 1L << 5;   // static at function definition

    //
    // Qualifiers for Fortran
    //
    public static final long TQ_FPUBLIC              = 1L << 6;   // public
    public static final long TQ_FPRIVATE             = 1L << 7;   // private
    public static final long TQ_FPROTECTED           = 1L << 8;   // protected
    public static final long TQ_FPOINTER             = 1L << 9;   // pointer
    public static final long TQ_FOPTIONAL            = 1L << 10;  // optional
    public static final long TQ_FTARGET              = 1L << 11;  // target
    public static final long TQ_FSAVE                = 1L << 12;  // save
    public static final long TQ_FPARAMETER           = 1L << 13;  // parameter
    public static final long TQ_FALLOCATABLE         = 1L << 14;  // allocatable
    public static final long TQ_FINTENT_IN           = 1L << 15;  // intent(in)
    public static final long TQ_FINTENT_OUT          = 1L << 16;  // intent(out)
    public static final long TQ_FINTENT_INOUT        = 1L << 17;  // intent(inout)
    public static final long TQ_FPROGRAM             = 1L << 18;  // program
    public static final long TQ_FINTRINSIC           = 1L << 19;  // intrinsic
    public static final long TQ_FELEMENTAL           = 1L << 20;  // elemental
    public static final long TQ_FIMPURE              = 1L << 21;  // impure
    public static final long TQ_FPURE                = 1L << 22;  // pure
    public static final long TQ_FRECURSIVE           = 1L << 23;  // recursive
    public static final long TQ_FINTERNAL            = 1L << 24;  // internal
    public static final long TQ_FEXTERNAL            = 1L << 25;  // external
    public static final long TQ_FSEQUENCE            = 1L << 26;  // sequence
    public static final long TQ_FINTERNAL_PRIVATE    = 1L << 27;  // private in structure decl
    public static final long TQ_FCRAY_POINTER        = 1L << 28;  // cray pointer (ID=060c)
    public static final long TQ_FVOLATILE            = 1L << 29;  // volatile
    public static final long TQ_FCLASS               = 1L << 30;  // class
    public static final long TQ_FVALUE               = 1L << 31;  // value
    public static final long TQ_FMODULE              = 1L << 32;  // module
    public static final long TQ_FPROCEDURE           = 1L << 33;  // procedure
    public static final long TQ_FCONTIGUOUS          = 1L << 34;  // contiguous
    public static final long TQ_FASYNCHRONOUS        = 1L << 35;  // asynchronous
    public static final long TQ_FABSTRACT            = 1L << 36;  // abstract
    public static final long TQ_FDEFERRED            = 1L << 37;  // deferred

    private String type_id;
    private int type_kind;

    /* Type Qualifiers */
    private long type_qual_flags;

    /** for marking */
    boolean is_marked;

    /** copy source */
    public Xtype copied;

    /** sequence fo generating type id */
    private static int gen_id_seq = 0;

    /** gcc attributes */
    private Xobject gcc_attrs;

    /** type name */
    protected Ident tag;

    /** coshape infos. incl. corank and codimensions (ID=060) */
    protected boolean is_coarray = false;
    protected Coshape coshape = new Coshape();

    /** parameterized derived type infos. */
    protected XobjList fTypeParamValues;

    /** ISO C binding information */
    protected String bind;
    protected String bind_name;

    /** Fortran procedure pointer pass attributes */
    private String pass;
    private String pass_arg_name;

    /*
     * for pre-defined basic type
     */
    public static final BasicType voidType =
        new BasicType(BasicType.VOID);
    public static final Xtype voidPtrType =
        Xtype.Pointer(voidType);
    public static final BasicType charType =
        new BasicType(BasicType.CHAR);
    public static final BasicType shortType =
        new BasicType(BasicType.SHORT);
    public static final BasicType intType =
        new BasicType(BasicType.INT);
    public static final BasicType longType =
        new BasicType(BasicType.LONG);
    public static final BasicType longlongType =
        new BasicType(BasicType.LONGLONG);
    public static final BasicType unsignedcharType =
        new BasicType(BasicType.UNSIGNED_CHAR);
    public static final BasicType unsignedshortType =
        new BasicType(BasicType.UNSIGNED_SHORT);
    public static final BasicType unsignedType =
        new BasicType(BasicType.UNSIGNED_INT);
    public static final BasicType unsignedlongType =
        new BasicType(BasicType.UNSIGNED_LONG);
    public static final BasicType unsignedlonglongType =
        new BasicType(BasicType.UNSIGNED_LONGLONG);
    public static final BasicType floatType =
        new BasicType(BasicType.FLOAT);
    public static final BasicType doubleType =
        new BasicType(BasicType.DOUBLE);
    public static final BasicType longdoubleType =
        new BasicType(BasicType.LONG_DOUBLE);
    public static final BasicType boolType =
        new BasicType(BasicType.BOOL);
    public static final BasicType gccBuiltinVaListType =
        new BasicType(BasicType.GCC_BUILTIN_VA_LIST);
    public static final BasicType floatComplexType =
        new BasicType(BasicType.FLOAT_COMPLEX);
    public static final BasicType doubleComplexType =
        new BasicType(BasicType.DOUBLE_COMPLEX);
    public static final BasicType longDoubleComplexType =
        new BasicType(BasicType.LONG_DOUBLE_COMPLEX);

    public static final BasicType FintType =
        intType;
    public static final BasicType Fint4Type =
        new BasicType(BasicType.INT, null, 0, null, Xcons.IntConstant(4), null);
    public static final BasicType Fint8Type =
        new BasicType(BasicType.INT, null, 0, null, Xcons.IntConstant(8), null);
    public static final BasicType FcharacterType =
        new BasicType(BasicType.F_CHARACTER);
    public static final BasicType FcharacterWithLenType =
        new BasicType(BasicType.F_CHARACTER, null, 0, null, null, Xcons.IntConstant(-1));
    public static final BasicType FlogicalType =
        boolType;
    public static final BasicType FrealType =
        floatType;
    public static final BasicType FcomplexType =
        floatComplexType;
    public static final BasicType FnumericType =
        new BasicType(BasicType.F_NUMERIC);
    public static final BasicType FnumericAllType =
        new BasicType(BasicType.F_NUMERIC_ALL);
    public static final BasicType FvoidType =
        voidType;
    public static final FunctionType FsubroutineType =
        new FunctionType(FvoidType);
    public static final FunctionType FintFunctionType =
        new FunctionType(FintType);
    public static final FunctionType FlogicalFunctionType =
        new FunctionType(FlogicalType);
    public static final FunctionType FexternalSubroutineType =
        new FunctionType(FvoidType, TQ_FEXTERNAL);
    public static final FunctionType FexternalLogicalFunctionType =
        new FunctionType(FlogicalType, TQ_FEXTERNAL);
    public static final FunctionType FexternalIntFunctionType =
        new FunctionType(FintType, TQ_FEXTERNAL);
    public static final FunctionType FnumericalAllFunctionType =
        new FunctionType(FnumericAllType);

    public static final Xtype stringType =
        Xtype.Pointer(Xtype.charType);
    public static final Xtype FuintPtrType =
        new BasicType(BasicType.INT, null, 0, null,
            Xcons.IntConstant(MachineDepConst.SIZEOF_VOID_P), null);
    public static final Xtype indvarType = new PointerType(voidType); // not Xtype.Pointer(voidType)
    public static final Xtype indvarPtrType =
        new PointerType(indvarType); // not Xtype.Pointer(indvarType)

    public static class TypeInfo
    {
        public final Xtype type;
        public final String cname, fname, fnamesub;

        TypeInfo(Xtype type, String cname, String fname, String fnamesub)
        {
            this.type = type;
            this.cname = cname;
            this.fname = fname;
            this.fnamesub = fnamesub;
        }
    }

    /** type mapping of Xtype, XcodeML/C, XcodeML/Fortran */
    protected static final TypeInfo[] type_infos =
    {
        new TypeInfo(voidType,              "void",                 "Fvoid",            "Fvoid"),
        new TypeInfo(charType,              "char",                 null,               "Fint"),
        new TypeInfo(unsignedcharType,      "unsigned_char",        null,               "Fint"),
        new TypeInfo(shortType,             "short",                null,               "Fint"),
        new TypeInfo(unsignedshortType,     "unsigned_short",       null,               "Fint"),
        new TypeInfo(intType,               "int",                  "Fint",             "Fint"),
        new TypeInfo(unsignedType,          "unsigned",             null,               "Fint"),
        new TypeInfo(longType,              "long",                 null,               "Fint"),
        new TypeInfo(unsignedlongType,      "unsigned_long",        null,               "Fint"),
        new TypeInfo(longlongType,          "long_long",            null,               "Fint"),
        new TypeInfo(unsignedlonglongType,  "unsigned_long_long",   null,               "Fint"),
        new TypeInfo(floatType,             "float",                "Freal",               "Freal"),
        new TypeInfo(doubleType,            "double",               null,            "Freal"),
        new TypeInfo(longdoubleType,        "long_double",          null,               "Freal"),
        new TypeInfo(boolType,              "bool",                 "Flogical",         "Flogical"),
        new TypeInfo(floatComplexType,      "float_complex",        "Fcomplex",         "Fcomplex"),
        new TypeInfo(doubleComplexType,     "double_complex",       null,               "Fcomplex"),
        new TypeInfo(longDoubleComplexType, "long_double_complex",  null,               "Fcomplex"),
        new TypeInfo(gccBuiltinVaListType,  "__builtin_va_list",    null,               null),
        new TypeInfo(FcharacterType,        null,                   "Fcharacter",       "Fcharacter"),
        new TypeInfo(FnumericType,          null,                   "Fnumeric",         "Fnumeric"),
        new TypeInfo(FnumericAllType,       null,                   "FnumericAll",      "FnumericAll"),
    };

    // constructor
    public Xtype(int kind, String id, long typeQualFlags, Xobject gccAttrs,
                 Ident tag, Xobject[] codimensions)
    {
        this.type_kind = kind;
        this.type_id = id;
        setTypeQualFlags(typeQualFlags);
        this.gcc_attrs = gccAttrs;
        this.tag = tag;
        setCodimensions(codimensions);
    }

    public void assign(Xtype op2)
    {
        this.type_id = op2.type_id;
        this.type_kind = op2.type_kind;
        this.type_qual_flags = op2.type_qual_flags;
        this.is_marked = op2.is_marked;
        this.copied = op2.copied;
        this.gen_id_seq = op2.gen_id_seq;
        this.gcc_attrs = op2.gcc_attrs;
        this.tag = op2.tag;
        this.fTypeParamValues = op2.fTypeParamValues;
        this.bind = op2.bind;
        this.bind_name = op2.bind_name;
        this.pass = op2.pass;
        this.pass_arg_name = op2.pass_arg_name;
        setCodimensions(op2.getCodimensions());
    }

    public Xtype(int kind, String id, long typeQualFlags, Xobject gccAttrs,
                 Ident tag)
    {
        this(kind, id, typeQualFlags, gccAttrs, tag, null);
    }

    public Xtype(int kind, String id, long typeQualFlags, Xobject gccAttrs,
                 Xobject[] codimensions)
    {
        this(kind, id, typeQualFlags, gccAttrs, null, codimensions);
    }

    public Xtype(int kind, String id, long typeQualFlags, Xobject gccAttrs)
    {
        this(kind, id, typeQualFlags, gccAttrs, null, null);
    }

    public Xtype(int kind)
    {
        this(kind, null, 0, null, null, null);
    }

    /** return if is qualifed by qualifier,
     * gcc attribute, Fortran kind parameter or Fortran len parameter
     */
    public final boolean isQualified()
    {
        return type_qual_flags != 0 ||
            gcc_attrs != null ||
            getFkind() != null ||
            getFlen() != null;
    }

    /** return type id */
    public final String Id()
    {
        return type_id;
    }

    /** return type id for XcodeML/C */
    public final String getXcodeCId()
    {
        return type_id != null ? type_id :
            ((copied != null) ? copied.getXcodeCId() :
                (isBasic() ? BasicType.getTypeInfo(getBasicType()).cname : null));
    }

    /** return type id for XcodeML/F */
    public final String getXcodeFId()
    {
        return type_id != null ? type_id :
            ((copied != null) ? copied.getXcodeFId() :
                (isBasic() ? BasicType.getTypeInfo(getBasicType()).fname : null));
    }

    /** set kind constant (BASIC, STRUCT, UNION, ...) */
    public final void setKind(int kind)
    {
        type_kind = kind;
    }

    /** get kind constant (BASIC, STRUCT, UNION, ...) */
    public final int getKind()
    {
        return type_kind;
    }

    /** set type qualifier flag (TQ_*). */
    private final void setTypeQualFlag(long flag, boolean enabled)
    {
        if(enabled)
            type_qual_flags |= flag;
        else
            type_qual_flags &= ~flag;
    }

    /** get type qualifier flags (TQ_*). */
    public final long getTypeQualFlags()
    {
        return type_qual_flags;
    }

    /** get type qualifier flag (TQ_*). */
    private final boolean getTypeQualFlag(long flag)
    {
        return (type_qual_flags & flag) > 0;
    }

    /** set type qualifier flags (TQ_*). */
    public final void setTypeQualFlags(long flags)
    {
        type_qual_flags = flags;
    }

    /** return if is qualified by 'const' */
    public final boolean isConst()
    {
        return getTypeQualFlag(TQ_CONST);
    }

    /** set qualifier 'const' */
    public final void setIsConst(boolean enabled)
    {
        setTypeQualFlag(TQ_CONST, enabled);
    }

    /** return if is qualified by 'volatile' */
    public final boolean isVolatile()
    {
        return getTypeQualFlag(TQ_VOLATILE);
    }

    /** set qualifier 'volatile' */
    public final void setIsVoaltile(boolean enabled)
    {
        setTypeQualFlag(TQ_VOLATILE, enabled);
    }

    /** return if is qualified by 'restrict' */
    public final boolean isRestrict()
    {
        return getTypeQualFlag(TQ_RESTRICT);
    }

    /** set qualifier 'restirct' */
    public final void setIsRestrict(boolean enabled)
    {
        setTypeQualFlag(TQ_RESTRICT, enabled);
    }

    /** return if is qualified by 'inline' */
    public final boolean isInline()
    {
        return getTypeQualFlag(TQ_INLINE);
    }

    /** set qualifier 'inline' */
    public final void setIsInline(boolean enabled)
    {
        setTypeQualFlag(TQ_INLINE, enabled);
    }

    /** return if is qualified by 'static' in array specifier */
    public final boolean isArrayStatic()
    {
        return getTypeQualFlag(TQ_ARRAY_STATIC);
    }

    /** set qualifier 'static' in array specifier */
    public final void setIsArrayStatic(boolean enabled)
    {
        setTypeQualFlag(TQ_ARRAY_STATIC, enabled);
    }

    /** return if is qualified by 'static' at function definition */
    public final boolean isFuncStatic()
    {
        return getTypeQualFlag(TQ_FUNC_STATIC);
    }

    /** set qualifier 'static' at function definition */
    public final void setIsFuncStatic(boolean enabled)
    {
        setTypeQualFlag(TQ_FUNC_STATIC, enabled);
    }

    /** Fortran : return if is qualified by 'public' */
    public final boolean isFpublic()
    {
        return getTypeQualFlag(TQ_FPUBLIC);
    }

    /** Fortran : set qualifier 'public' */
    public final void setIsFpublic(boolean enabled)
    {
        setTypeQualFlag(TQ_FPUBLIC, enabled);
    }

    /** Fortran : return if is qualified by 'private' */
    public final boolean isFprivate()
    {
        return getTypeQualFlag(TQ_FPRIVATE);
    }

    /** Fortran : set qualifier 'private' */
    public final void setIsFprivate(boolean enabled)
    {
        setTypeQualFlag(TQ_FPRIVATE, enabled);
    }

    /** Fortran : return if is qualified by 'protected' */
    public final boolean isFprotected()
    {
        return getTypeQualFlag(TQ_FPROTECTED);
    }

    /** Fortran : set qualifier 'protected' */
    public final void setIsFprotected(boolean enabled)
    {
        setTypeQualFlag(TQ_FPROTECTED, enabled);
    }
    
    /** Fortran : return if is qualified by 'pointer' */
    public final boolean isFpointer()
    {
        return getTypeQualFlag(TQ_FPOINTER);
    }

    /** Fortran : set qualifier 'pointer' */
    public final void setIsFpointer(boolean enabled)
    {
        setTypeQualFlag(TQ_FPOINTER, enabled);
    }

    /** Fortran : return if is qualified by 'optional' */
    public final boolean isFoptional()
    {
        return getTypeQualFlag(TQ_FOPTIONAL);
    }

    /** Fortran : set qualifier 'optional' */
    public final void setIsFoptional(boolean enabled)
    {
        setTypeQualFlag(TQ_FOPTIONAL, enabled);
    }

    /** Fortran : return if is qualified by 'target' */
    public final boolean isFtarget()
    {
        return getTypeQualFlag(TQ_FTARGET);
    }

    /** Fortran : set qualifier 'target' */
    public final void setIsFtarget(boolean enabled)
    {
        setTypeQualFlag(TQ_FTARGET, enabled);
    }

    /** Fortran : return if is qualified by 'save' */
    public final boolean isFsave()
    {
        return getTypeQualFlag(TQ_FSAVE);
    }

    /** Fortran : set qualifier 'save' */
    public final void setIsFsave(boolean enabled)
    {
        setTypeQualFlag(TQ_FSAVE, enabled);
    }

    /** Fortran : return if is qualified by 'parameter' */
    public final boolean isFparameter()
    {
        return getTypeQualFlag(TQ_FPARAMETER);
    }

    /** Fortran : set qualifier 'parameter' */
    public final void setIsFparameter(boolean enabled)
    {
        setTypeQualFlag(TQ_FPARAMETER, enabled);
    }

    /** Fortran : return if is qualified by 'allocatable' */
    public final boolean isFallocatable()
    {
        return getTypeQualFlag(TQ_FALLOCATABLE);
    }

    /** Fortran : set qualifier 'allocatable' */
    public final void setIsFallocatable(boolean enabled)
    {
        setTypeQualFlag(TQ_FALLOCATABLE, enabled);
    }

    /** Fortran : return if it is a cray pointer (ID=060c) */
    public final boolean isFcrayPointer()
    {
        return getTypeQualFlag(TQ_FCRAY_POINTER);
    }

    /** Fortran : return if it is volatile */
    public final boolean isFvolatile()
    {
        return getTypeQualFlag(TQ_FVOLATILE);
    }

    /** Fortran : set qualifier 'cray pointer' (ID=060c) */
    public final void setIsFcrayPointer(boolean enabled)
    {
        setTypeQualFlag(TQ_FCRAY_POINTER, enabled);
    }

    /** Fortran : return if is qualified by 'intent(in)' */
    public final boolean isFintentIN()
    {
        return getTypeQualFlag(TQ_FINTENT_IN);
    }

    /** Fortran : set qualifier 'intent(in)' */
    public final void setIsFintentIN(boolean enabled)
    {
        setTypeQualFlag(TQ_FINTENT_IN, enabled);
    }

    /** Fortran : return if is qualified by 'intent(out)' */
    public final boolean isFintentOUT()
    {
        return getTypeQualFlag(TQ_FINTENT_OUT);
    }

    /** Fortran : set qualifier 'intent(out)' */
    public final void setIsFintentOUT(boolean enabled)
    {
        setTypeQualFlag(TQ_FINTENT_OUT, enabled);
    }

    /** Fortran : return if is qualified by 'intent(inout)' */
    public final boolean isFintentINOUT()
    {
        return getTypeQualFlag(TQ_FINTENT_INOUT);
    }

    /** Fortran : set qualifier 'intent(inout)' */
    public final void setIsFintentINOUT(boolean enabled)
    {
        setTypeQualFlag(TQ_FINTENT_INOUT, enabled);
    }

    /** Fortran : return if is qualified by 'program' */
    public final boolean isFprogram()
    {
        return getTypeQualFlag(TQ_FPROGRAM);
    }

    /** Fortran : set qualifier 'program' */
    public final void setIsFprogram(boolean enabled)
    {
        setTypeQualFlag(TQ_FPROGRAM, enabled);
    }

    /** Fortran : return if is qualified by 'intrinsic' */
    public final boolean isFintrinsic()
    {
        return getTypeQualFlag(TQ_FINTRINSIC);
    }

    /** Fortran : set qualifier 'intrinsic' */
    public final void setIsFintrinsic(boolean enabled)
    {
        setTypeQualFlag(TQ_FINTRINSIC, enabled);
    }

    /** Fortran : return if is qualified by 'elemental' */
    public final boolean isFelemental()
    {
        return getTypeQualFlag(TQ_FELEMENTAL);
    }

    /** Fortran : set qualifier 'elemental' */
    public final void setIsFelemental(boolean enabled)
    {
        setTypeQualFlag(TQ_FELEMENTAL, enabled);
    }

    /** Fortran : return if is qualified by 'impure' */
    public final boolean isFimpure()
    {
        return getTypeQualFlag(TQ_FIMPURE);
    }

    /** Fortran : set qualifier 'impure' */
    public final void setIsFimpure(boolean enabled)
    {
        setTypeQualFlag(TQ_FIMPURE, enabled);
    }

    /** Fortran : return if is qualified by 'pure' */
    public final boolean isFpure()
    {
        return getTypeQualFlag(TQ_FPURE);
    }

    /** Fortran : set qualifier 'pure' */
    public final void setIsFpure(boolean enabled)
    {
        setTypeQualFlag(TQ_FPURE, enabled);
    }

    /** Fortran : return if is qualified by 'recursive' */
    public final boolean isFrecursive()
    {
        return getTypeQualFlag(TQ_FRECURSIVE);
    }

    /** Fortran : set qualifier 'recursive' */
    public final void setIsFrecursive(boolean enabled)
    {
        setTypeQualFlag(TQ_FRECURSIVE, enabled);
    }

    /** Fortran : return if is qualified by 'internal' */
    public final boolean isFinternal()
    {
        return getTypeQualFlag(TQ_FINTERNAL);
    }

    /** Fortran : set qualifier 'internal' */
    public final void setIsFinternal(boolean enabled)
    {
        setTypeQualFlag(TQ_FINTERNAL, enabled);
    }

    /** Fortran : return if is qualified by 'external' */
    public final boolean isFexternal()
    {
        return getTypeQualFlag(TQ_FEXTERNAL);
    }

    /** Fortran : set qualifier 'external' */
    public final void setIsFexternal(boolean enabled)
    {
        setTypeQualFlag(TQ_FEXTERNAL, enabled);
    }

    /** Fortran : return if is qualified by 'sequence' */
    public final boolean isFsequence()
    {
        return getTypeQualFlag(TQ_FSEQUENCE);
    }

    /** Fortran : set qualifier 'sequence' */
    public final void setIsFsequence(boolean enabled)
    {
        setTypeQualFlag(TQ_FSEQUENCE, enabled);
    }

    /** Fortran : return if is qualified by 'private' in structure decl */
    public final boolean isFinternalPrivate()
    {
        return getTypeQualFlag(TQ_FINTERNAL_PRIVATE);
    }

    /** Fortran : set qualifier 'private' in structure decl */
    public final void setIsFinternalPrivate(boolean enabled)
    {
        setTypeQualFlag(TQ_FINTERNAL_PRIVATE, enabled);
    }

    /** Fortran : return if is qualified by 'class' in pointer decl */
    public final boolean isFclass()
    {
        return getTypeQualFlag(TQ_FCLASS);
    }

    /** Fortran : set qualifier 'class' in pointer decl */
    public final void setIsFclass(boolean enabled)
    {
        setTypeQualFlag(TQ_FCLASS, enabled);
    }

    /** Fortran : return if is qualified by 'value' in pointer decl */
    public final boolean isFvalue()
    {
      return getTypeQualFlag(TQ_FVALUE);
    }

    /** Fortran : set qualifier 'value' in pointer decl */
    public final void setIsFvalue(boolean enabled)
    {
        setTypeQualFlag(TQ_FVALUE, enabled);
    }

    /** Fortran : return if is qualified by 'module' in subroutine/function decl */
    public final boolean isFmodule()
    {
      return getTypeQualFlag(TQ_FMODULE);
    }

    /** Fortran : set qualifier 'module' in subroutine/function decl */
    public final void setIsFmodule(boolean enabled)
    {
        setTypeQualFlag(TQ_FMODULE, enabled);
    }

    /** Fortran : return if is 'procedure' decl */
    public final boolean isFprocedure()
    {
      return getTypeQualFlag(TQ_FPROCEDURE);
    }

    /** Fortran : set attribute 'is_procedure' */
    public final void setIsFprocedure(boolean enabled)
    {
        setTypeQualFlag(TQ_FPROCEDURE, enabled);
    }

    /** Fortran : return if is contiguous */
    public final boolean isFcontiguous()
    {
      return getTypeQualFlag(TQ_FCONTIGUOUS);
    }

    /** Fortran : set attribute 'is_contiguous' */
    public final void setIsFcontiguous(boolean enabled)
    {
        setTypeQualFlag(TQ_FCONTIGUOUS, enabled);
    }

    /** Fortran : return if is asynchronous */
    public final boolean isFasynchronous()
    {
      return getTypeQualFlag(TQ_FASYNCHRONOUS);
    }

    /** Fortran : set attribute 'is_asynchronous' */
    public final void setIsFasynchronous(boolean enabled)
    {
        setTypeQualFlag(TQ_FASYNCHRONOUS, enabled);
    }

    /** Fortran : return if is abstract */
    public final boolean isFabstract()
    {
      return getTypeQualFlag(TQ_FABSTRACT);
    }

    /** Fortran : set attribute 'is_abstract' */
    public final void setIsFabstract(boolean enabled)
    {
        setTypeQualFlag(TQ_FABSTRACT, enabled);
    }

    /** Fortran : return if is dererred */
    public final boolean isFdeferred()
    {
      return getTypeQualFlag(TQ_FDEFERRED);
    }

    /** Fortran : set attribute 'is_deferred' */
    public final void setIsFdeferred(boolean enabled)
    {
        setTypeQualFlag(TQ_FDEFERRED, enabled);
    }

    /** Fortran : return if is qualified by 'bind' in pointer decl */
    public final String getBind()
    {
      return bind;
    }

    /** Fortran : return if is qualified by 'bind_name' in pointer decl */
    public final String getBindName()
    {
      return bind_name;
    }

    /** Fortran : set qualifier 'bind' in pointer decl */
    public final void setBind(String value)
    {
        bind = value;
    }

    /** Fortran : set qualifier 'bind_name' in pointer decl */
    public final void setBindName(String value)
    {
        bind_name = value;
    }

    /** get basic type kind (BasicType.*) */
    public int getBasicType()
    {
        throw new UnsupportedOperationException();
    }

    /** set reference type */
    public void setRef(Xtype ref)
    {
        throw new UnsupportedOperationException();
    }

    /** set get type */
    public Xtype getRef()
    {
        throw new UnsupportedOperationException();
    }

    /** return if is function prototype */
    public boolean isFuncProto()
    {
        throw new UnsupportedOperationException();
    }

    /** get function parameters */
    public Xobject getFuncParam()
    {
        throw new UnsupportedOperationException();
    }

    /** Fortran: get function result name */
    public String getFuncResultName()
    {
        throw new UnsupportedOperationException();
    }

    /** Fortran: set function result name */
    public void setFuncResultName(String fresult_name)
    {
        throw new UnsupportedOperationException();
    }

    @Deprecated
    public final long getArrayDim()
    {
        return getArraySize();
    }

    /** C: get fixed array size. -1 means variabel array size */
    public long getArraySize()
    {
        throw new UnsupportedOperationException();
    }

    /** C: return if is not set array size */
    public boolean isNoArraySize()
    {
        return isArray() && getArraySize() == -1 && getArraySizeExpr() == null;
    }

    /** C: return if is variable array size */
    public boolean isVariableArray()
    {
        return isArray() && getArraySize() == -1 && getArraySizeExpr() != null;
    }

    @Deprecated
    public final Xobject getArrayAdjSize()
    {
        return getArraySizeExpr();
    }

    /** C: get variable array size expression */
    public Xobject getArraySizeExpr()
    {
        throw new UnsupportedOperationException();
    }

    /** Fortarn: get Fortran array size expressions */
    public Xobject[] getFarraySizeExpr()
    {
        throw new UnsupportedOperationException();
    }

    /** Fortran: get Fortran array size or 1 for scalar */     // #060
    public Xobject getTotalArraySizeExpr(Block block)
    {
        throw new UnsupportedOperationException();
    }

    /** Fortran: get Fortran type element length (bytes) in Expr */     // #060
    public Xobject getElementLengthExpr(Block block)
    {
        throw new UnsupportedOperationException();
    }

    /** Fortran: get Fortran type element length (bytes) in integer */     // #060
    public int getElementLength(Block block)
    {
        throw new UnsupportedOperationException();
    }

    /** Fortran: return if is assumed size array */
    public boolean isFassumedSize()
    {
        return false;
    }

    /** Fortran: return if is assumed shape array */
    public boolean isFassumedShape()
    {
        return false;
    }

    /** Fortran: return if is fixed size array */
    public boolean isFfixedShape()
    {
        return false;
    }

    /** Fortran: convert Fortran array size */
    public void convertFindexRange(boolean extendsLowerBound, boolean extendsUpperBound, Block b)
    {
        throw new UnsupportedOperationException();
    }

    /** Fortran: convert Fortran array size */
    public void convertToAssumedShape()
    {
        throw new UnsupportedOperationException();
    }

    /** get number of array dimensions */
    public int getNumDimensions()
    {
        return 0;
    }

    /** get array element type */
    public Xtype getArrayElementType()
    {
        throw new UnsupportedOperationException();
    }

    /*
     *  implements Coshape (#060)
     */
    public int getCorank()
    {
        return coshape.getCorank();
    }
    public boolean isCoarray()
    {
        return is_coarray;
    }
    public void setIsCoarray(boolean is_coarray)
    {
        this.is_coarray = is_coarray;
    }
    public boolean wasCoarray()
    {
        return getCorank() > 0;
    }
    public Xobject[] getCodimensions()
    {
        return coshape.getCodimensions();
    }
    public void setCodimensions(Xobject[] codimensions)
    {
        coshape.setCodimensions(codimensions);
        is_coarray  = (getCorank() > 0);
    }
    public void setFTypeParamValues(XobjList params)
    {
        fTypeParamValues = params;
    }
    public XobjList getFTypeParamValues()
    {
        return fTypeParamValues;
    }
    public void removeCodimensions()
    {
        coshape.removeCodimensions();
        is_coarray  = false;
    }
    public void hideCodimensions()
    {
        is_coarray  = false;
    }
     public Xobject[] copyCodimensions()
    {
        return coshape.copyCodimensions();
    }

    /** @deprecated */
    @Deprecated
    public XobjList getMoeList()
    {
        return getMemberList();
    }

    /** get composite type member list */
    public XobjList getMemberList()
    {
        throw new UnsupportedOperationException();
    }

    /** get composite type proc list */
    public XobjList getProcList()
    {
        throw new UnsupportedOperationException();
    }

    /** get type of member which has specified name */
    public Xtype getMemberType(String member)
    {
        throw new UnsupportedOperationException();
    }

    /** create copy */
    public final Xtype copy()
    {
    	return copy(null);
    }

    /** create copy */
    protected Xtype copy(String id)
    {
        return new Xtype(type_kind, id, getTypeQualFlags(), gcc_attrs, tag, copyCodimensions());
    }

    /** create copy. created copy references this instance as copied */
    public Xtype inherit(String id)
    {
    	Xtype t = copy(id);
    	t.copied = this;
    	return t;
    }

    /** return if is function */
    public final boolean isFunction()
    {
        return type_kind == FUNCTION;
    }

    /** Fortran: return if is subroutine */
    public final boolean isFsubroutine()
    {
        return isFunction() && getRef() == Xtype.FvoidType;
    }

    /** C: return if is pointer */
    public final boolean isPointer()
    {
        return (type_kind == POINTER || type_kind == ARRAY);
    }

    /** return if is basic type */
    public final boolean isBasic()
    {
        return type_kind == BASIC;
    }

    /** C: return if is array */
    public final boolean isArray()
    {
        return type_kind == ARRAY;
    }

    /** Fortran: return if is array */
    public final boolean isFarray()
    {
        return type_kind == F_ARRAY;
    }

    /** return if is struct/type */
    public final boolean isStruct()
    {
        return type_kind == STRUCT;
    }

    /** C: return if is union */
    public final boolean isUnion()
    {
        return type_kind == UNION;
    }

    /** C: return if is enum */
    public final boolean isEnum()
    {
        return type_kind == ENUM;
    }

    /** C: return if is union */
    public final boolean isFcharacter()
    {
        return type_kind == BASIC && getBasicType() == BasicType.F_CHARACTER;
    }

    /** C: return if is unsigned type */
    public boolean isUnsigned()
    {
        return false;
    }

    /** return if is integral type */
    public boolean isIntegral()
    {
        return false;
    }

    /** return if is bool type */
    public boolean isBool()
    {
        return false;
    }

    /** return if is float type */
    public boolean isFloating()
    {
        return false;
    }

    /** return if is complex or imaginary type */
    public boolean isComplexOrImaginary()
    {
        return false;
    }

    /** return if is numeric type */
    //public final boolean isNumeric()       #357
    public boolean isNumeric()
    {
        return isIntegral() || isFloating() || isComplexOrImaginary();
    }

    /** return if is void type */
    public boolean isVoid()
    {
        return false;
    }

    /** Fortran: get kind parameter */
    public Xobject getFkind()
    {
        return null;
    }

    /** Fortran: get len parameter */
    public Xobject getFlen()
    {
        return null;
    }

    /** Fortran: get procedure pointer pass attribute */
    public String getPass()
    {
        return pass;
    }

    /** Fortran: get procedure pointer pass attribute */
    public void setPass(String pass)
    {
        this.pass = pass;
    }

    /** Fortran: get procedure pointer pass arg name attribute */
    public String getPassArgName()
    {
        return pass_arg_name;
    }

    /** Fortran: get procedure pointer pass arg name attribute */
    public void setPassArgName(String pass_arg_name)
    {
        this.pass_arg_name = pass_arg_name;
    }

    /** Fortran: return if len parameter is variable value */
    public boolean isFlenVariable()
    {
        return false;
    }

    /** Fortran: return if len parameter is variable value */
    public boolean isFlenAssumedShape()
    {
        return false;
    }

    /** Fortran: return if len parameter is variable value */
    public boolean isFlenAssumedSize()
    {
        return false;
    }

    /** Fortran: return if the type extends parent type */
    public boolean isExtended()
    {
        return false;
    }

    /** return is equals specified type */
    public final boolean equals(Xtype t)
    {
    	if(t == null)
    		return false;
        if(t == this)
            return true;
        if(type_kind != t.type_kind)
            return false;
        switch(type_kind) {
        case BASIC:
            return this.getBasicType() == t.getBasicType();
        case ENUM:
        	return ((t.getTypeQualFlags() == getTypeQualFlags()) &&
    			(t.getGccAttributes() == getGccAttributes()) &&
    			(((EnumType)t).getMemberList() == ((EnumType)this).getMemberList()));
        case STRUCT:
        case UNION:
        	return ((t.getTypeQualFlags() == getTypeQualFlags()) &&
    			(t.getGccAttributes() == getGccAttributes()) &&
    			(((CompositeType)t).getMemberList() == ((CompositeType)this).getMemberList()));
        case ARRAY:
            if(this.getArraySize() != t.getArraySize())
                return false;
        case FUNCTION:
        case POINTER:
            return this.getRef().equals(t.getRef());
        }
        return false;
    }

    @Override
    public String toString()
    {
        if(Id() == null) {
            String p = null;
            Xtype ref = null;
            switch(getKind()) {
            case Xtype.BASIC:
                p = "B";
                ref = copied;
                if(ref == null) {
                    String name = BasicType.getTypeInfo(getBasicType()).cname;
                    if(name == null)
                        name = BasicType.getTypeInfo(getBasicType()).fname;
                    if(getFkind() == null)
                        return name;
                    if(getFkind().Opcode() == Xcode.INT_CONSTANT)
                        return name + "(kind=" + getFkind().getInt() + ")";
                    return name + "(kind=" + getFkind().toString() + ")";
                }
                break;
            case Xtype.FUNCTION:    p = "F"; break;
            case Xtype.ARRAY:       p = "A"; ref = getRef(); break;
            case Xtype.POINTER:     p = "P"; ref = getRef(); break;
            case Xtype.STRUCT:      p = "S"; break;
            case Xtype.UNION:       p = "U"; break;
            case Xtype.ENUM:        p = "E"; break;
            case Xtype.F_ARRAY:     p = "FA"; break;
            }
            return "<" + p + (ref != null ? ":" + ref.toString() : "")+">";
        }

        String id = getXcodeCId();
        if(id == null)
            id = getXcodeFId();
        return id;
    }

    //
    // static constructor
    //

    /**
     * C: create pointer type.
     * Fortran: return ref.
     */
    public static Xtype Pointer(Xtype ref)
    {
        if(XmOption.isLanguageF())
            return ref;
        return new PointerType(ref);
    }

    /**
     * C: create void pointer type.
     * Fortran: integer type whose size equals to void pointer.
     */
    public static Xtype VoidPtrOrFuintPtr()
    {
        if(XmOption.isLanguageC())
            return Pointer(Xtype.voidType);
        return Xtype.FuintPtrType;
    }

    /**
     * C: create pointer type and add to type table.
     * Fortran: return ref.
     */
    public static Xtype Pointer(Xtype ref, XobjectFile objFile)
    {
        if(XmOption.isLanguageF())
            return ref;
        PointerType type = new PointerType(ref);
        type.generateId();
        objFile.addType(type);
        return type;
    }

    /** create function type */
    public static FunctionType Function(Xtype ref)
    {
        return new FunctionType(ref);
    }

    /** C: create array type */
    public static ArrayType Array(Xtype ref, long size)
    {
        return new ArrayType(ref, size);
    }

    /** C: create array type */
    public static ArrayType Array(Xtype ref, Xobject sizeExpr)
    {
        return new ArrayType(ref, sizeExpr);
    }

    /** Fortran: create array type */
    public static FarrayType Farray(Xtype ref, Xobject ... sizeExprs)
    {
        return new FarrayType(null, ref, 0, sizeExprs);
    }

    /** generate type ID and set it to self */
    public void generateId()
    {
        String prefix = null;

        switch(type_kind) {
        case BASIC:
            prefix = "B"; break;
        case ENUM:
            prefix = "E"; break;
        case STRUCT:
            prefix = "S"; break;
        case UNION:
            prefix = "U"; break;
        case FUNCTION:
            prefix = "F"; break;
        case ARRAY:
            prefix = "A"; break;
        case POINTER:
            prefix = "P"; break;
        case F_ARRAY:
            prefix = "R"; break;
        default:
            throw new IllegalStateException();
        }

        type_id = prefix + "X" + Integer.toHexString(++gen_id_seq);
    }

    /** get GCC attributes */
    public Xobject getGccAttributes()
    {
        return gcc_attrs;
    }

    /** set GCC attributes */
    public void setGccAttributes(Xobject gccAttrs)
    {
        gcc_attrs = gccAttrs;
    }

    /** get name represents type kind */
    public static String getKindName(int kind)
    {
        return kind_names[kind];
    }

    /** Fortran: unset save attribute */
    public void unsetIsFsave()
    {
        setIsFsave(false);
        if(isFarray() && getRef().isFsave()) {
            Xtype rt = getRef().copy();
            rt.unsetIsFsave();
            setRef(rt);
        } else if(copied != null && copied.isFsave()) {
        	copied = null;
        }
    }

    /** Fortran: unset target attribute */
    public void unsetIsFtarget()
    {
        setIsFtarget(false);
        if(isFarray() && getRef().isFtarget()) {
            Xtype rt = getRef().copy();
            rt.unsetIsFtarget();
            setRef(rt);
        } else if(copied != null && copied.isFtarget()) {
        	copied = null;
        }
    }

    /** get copy source */
    public Xtype getBaseRefType()
    {
        if(copied != null)
            return copied;
        return this;
    }

    /** get original reference */
    public Xtype getOriginal()
    {
    	return null;
    }

    /** Fortran: set type name identifier */
    public void setTagIdent(Ident tag)
    {
    	this.tag = tag;
    }

    /** Fortran: get type name identifier */
    public Ident getTagIdent()
    {
    	if(tag == null && copied != null)
    		return copied.getTagIdent();
    	return tag;
    }

    public String getTagName() {
      Ident id = getTagIdent();
      return id == null ? null : id.getName();
    }
}
