/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import exc.block.Block;
import exc.util.XobjectVisitable;
import exc.util.XobjectVisitor;
import xcodeml.IXobject;
import xcodeml.XmException;
import xcodeml.util.XmOption;


/**
 * Abstract class of expression tree objects in Xobject.
 */
public class Xobject extends PropObject implements IXobject, XobjectVisitable, IVarContainer
{
    Xcode code;
    Xtype type;
    protected IXobject parent;
    private VarScope scope;

    private int optional_flags;
    
    public static final int OPT_GCC_THREAD          = 1 << 0;
    public static final int OPT_GCC_EXTENSION       = 1 << 1;
    public static final int OPT_GCC_SYNTAX          = 1 << 2;
    public static final int OPT_SYNTAX_MODIFIED     = 1 << 3;
    public static final int OPT_PARSED              = 1 << 4;
    public static final int OPT_USED_IN_ARRAY_SIZE  = 1 << 5;
    public static final int OPT_TO_BE_FCOMMON       = 1 << 6;
    public static final int OPT_DELAYED_DECL        = 1 << 7;
    public static final int OPT_GLOBAL              = 1 << 8;
    public static final int OPT_ATOMIC_STMT         = 1 << 9;
    public static final int OPT_F_MODULE_VAR        = 1 << 10;
    public static final int OPT_INDUCTION_VAR       = 1 << 11;
    public static final int OPT_REWRITTED_XMP       = 1 << 12;
    
    /**
     * Constructs an Xobject with code and type. This constructor is usually
     * called superclass.
     */
    public Xobject(Xcode code, Xtype type, int optional_flags)
    {
        this.code = code;
        this.type = type;
        this.optional_flags = optional_flags;
    }

    public Xobject(Xcode code, Xtype type)
    {
        this(code, type, 0);
    }

    /** Returns the code of this Xobject */
    public final Xcode Opcode()
    {
        return code;
    }

    /** Returns the type of this Xobject */
    public final Xtype Type()
    {
        return type;
    }

    /** Sets the type of this Xobject */
    public final void setType(Xtype type)
    {
        this.type = type;
    }
    
    public final int getOptionalFlags()
    {
        return optional_flags;
    }
    
    public final void setOptionalFlags(int optional_flags)
    {
        this.optional_flags = optional_flags;
    }
    
    protected void setOptionalFlag(int flag, boolean enabled)
    {
        if(enabled)
            optional_flags |= flag;
        else
            optional_flags &= ~flag;
    }
    
    protected boolean getOptionalFlag(int flag)
    {
        return (optional_flags & flag) > 0;
    }
    
    public final boolean isGccThread()
    {
        return getOptionalFlag(OPT_GCC_THREAD);
    }
    
    public final void setIsGccThread(boolean enabled)
    {
        setOptionalFlag(OPT_GCC_THREAD, enabled);
    }
    
    public final boolean isGccExtension()
    {
        return getOptionalFlag(OPT_GCC_EXTENSION);
    }
    
    public final void setIsGccExtension(boolean enabled)
    {
        setOptionalFlag(OPT_GCC_EXTENSION, enabled);
    }
    
    public final boolean isGccSyntax()
    {
        return getOptionalFlag(OPT_GCC_SYNTAX);
    }

    public final void setIsGccSyntax(boolean enabled)
    {
        setOptionalFlag(OPT_GCC_SYNTAX, enabled);
    }
    
    public final boolean isSyntaxModified()
    {
        return getOptionalFlag(OPT_SYNTAX_MODIFIED);
    }
    
    public final void setIsSyntaxModified(boolean enabled)
    {
        setOptionalFlag(OPT_SYNTAX_MODIFIED, enabled);
    }
    
    public final boolean isParsed()
    {
        return getOptionalFlag(OPT_PARSED);
    }
    
    public final void setIsParsed(boolean enabled)
    {
        setOptionalFlag(OPT_PARSED, enabled);
    }
    
    public final boolean isToBeFcommon()
    {
        return getOptionalFlag(OPT_TO_BE_FCOMMON);
    }
    
    public final void setIsToBeFcommon(boolean enabled)
    {
        setOptionalFlag(OPT_TO_BE_FCOMMON, enabled);
    }
    
    public final boolean isDelayedDecl()
    {
        return getOptionalFlag(OPT_DELAYED_DECL);
    }
    
    public final void setIsDelayedDecl(boolean enabled)
    {
        setOptionalFlag(OPT_DELAYED_DECL, enabled);
    }
    
    public final boolean isCglobalVarOrFvar()
    {
        return XmOption.isLanguageF() || getOptionalFlag(OPT_GLOBAL);
    }
    
    public final void setIsGlobal(boolean enabled)
    {
        setOptionalFlag(OPT_GLOBAL, enabled);
    }
    
    public final boolean isAtomicStmt()
    {
        return getOptionalFlag(OPT_ATOMIC_STMT);
    }
    
    public final void setIsAtomicStmt(boolean enabled)
    {
        setOptionalFlag(OPT_ATOMIC_STMT, enabled);
    }
    
    public final boolean isFmoduleVar()
    {
        return getOptionalFlag(OPT_F_MODULE_VAR);
    }
    
    public final void setIsFmoduleVar(boolean enabled)
    {
        setOptionalFlag(OPT_F_MODULE_VAR, enabled);
    }
    
    public final boolean isInductionVar()
    {
        return getOptionalFlag(OPT_INDUCTION_VAR);
    }
    
    public final void setIsInductionVar(boolean enabled)
    {
        setOptionalFlag(OPT_INDUCTION_VAR, enabled);
    }
    
    public final boolean isRewrittedByXmp()
    {
        return getOptionalFlag(OPT_REWRITTED_XMP);
    }
    
    public final void setIsRewrittedByXmp(boolean enabled)
    {
        setOptionalFlag(OPT_REWRITTED_XMP, enabled);
    }
    
    public VarScope getScope()
    {
        return scope;
    }
    
    public final void setScope(VarScope scope)
    {
        this.scope = scope;
    }
    
    public final boolean isScopeGlobal()
    {
        return scope == VarScope.GLOBAL;
    }
    
    public final boolean isScopeLocal()
    {
        return scope == VarScope.LOCAL;
    }
    
    public final boolean isScopeParam()
    {
        return scope == VarScope.PARAM;
    }
    
    /**
     * Returns integer value in XobjInt.
     * Dummy method at base class, it causes exception.
     */
    public int getInt()
    {
        throw new UnsupportedOperationException(toString());
    }
    
    public boolean canGetInt()
    {
        return false;
    }

    /**
     * Returns string in XobjString.
     * Dummy method at base class, it causes exception.
     */
    public String getString()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Returns Symbol name in XobjString. it is equal to getString.
     * Dummy method at base class, it causes exception.
     */
    public String getSym()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Returns Name of Xobject.
     * Dummy method at base class, it causes exception.
     */
    public String getName()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Returns Name of Xobject.
     * Dummy method at base class, it causes exception.
     */
    public void setName(String name)
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Returns floating value if XobjLong.
     * Dummy method at base class, it causes exception.
     */
    public double getFloat()
    {
        throw new UnsupportedOperationException(toString());
    }
    
    public String getFloatString()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Returns long value (64bits) if XobjLong.
     * Dummy method at base class, it causes exception.
     */
    public long getLong()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Returns high part of long value if XobjLong.
     * Dummy method at base class, it causes exception.
     */
    public long getLongHigh()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Returns low part of long value if XobjLong.
     * Dummy method at base class, it causes exception.
     */
    public long getLongLow()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Get rank in the term of Fortran.
     * Eg: Rank of a scalar expr is 0.
     *     Rank of a subarray is equal to or less than the rank of
     *     the host array.
     */
    public int getFrank()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Return argument list in XobjArgs.
     * Dummy method at base class, it causes exception.
     */
    public Xobject getArg(int i)
    {
        throw new UnsupportedOperationException(toString());
    }
    
    /**
     * Get the i-th argument or null if no i-th argument.
     * Dummy method at base class, it causes exception.
     */
    public Xobject getArgOrNull(int i)
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Set argument list in XobjArg.
     * Dummy method at base class, it causes exception.
     */
    public void setArg(int i, Xobject x)
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Get number of argument in XobjArgs.
     * Dummy method at base class, it causes exception.
     */
    public int Nargs()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Get argument of 1 as XobjList.
     */
    public XobjList getIdentList()
    {
        return (XobjList)getArgOrNull(1);
    }
    
    /**
     * Returns true if no argument in XobjArgs.
     * Dummy method at base class, it causes exception.
     */
    public boolean isEmpty()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Get an argument list XobjArgs.
     * Dummy method at base class, it causes exception.
     */
    public XobjArgs getArgs()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Set an argument list in XobjArgs.
     * Dummy method at base class, it causes exception.
     */
    public void setArgs(XobjArgs l)
    {
        throw new UnsupportedOperationException(toString());
    }
    
    /**
     * Remove the specified argument in XobjArgs
     * Dummy method at base class, it causes exception.
     */
    public Xobject removeArgs(XobjArgs a)
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Remove the first argument in XobjArgs
     * Dummy method at base class, it causes exception.
     */
    public Xobject removeFirstArgs()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * Remove the last argument in XobjArgs
     * Dummy method at base class, it causes exception.
     */
    public Xobject removeLastArgs()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public Xobject operand()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public Xobject left()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public Xobject right()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public void setOperand(Xobject x)
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public void setLeft(Xobject x)
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public void setRight(Xobject x)
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public void add(Xobject a)
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public Xobject getTail()
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public void insert(Xobject a)
    {
        throw new UnsupportedOperationException(toString());
    }
    
    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public Xobject copy()
    {
        throw new UnsupportedOperationException(toString());
    }
    
    protected Xobject copyTo(Xobject o)
    {
        o.optional_flags = optional_flags;
        o.parent = parent;
        o.scope = scope;
        return o;
    }


    /**
     * 
     * Dummy method at base class, it causes exception.
     */
    public void setLineNo(LineNo ln)
    {
        throw new UnsupportedOperationException(toString());
    }

    /**
     * 
     * Dummy method at base class, it returns null.
     */
    public LineNo getLineNo()
    {
        return null;
    }

    /**
     * check the equality of two objects
     */
    @Override
    public boolean equals(Object x)
    {
        if(!(x instanceof Xobject))
            return false;
        return equals((Xobject)x);
    }

    public boolean equals(Xobject x)
    {
        if(this == x)
            return true;
        return (code == x.code && (type == x.type || type.equals(x.type)));
    }

    public final String OpcodeName()
    {
        if(code == null)
            return null;
        
        if(type == null)
            return code.toString();
        else
	  return code.toString() + ":" + type.toString();
    }

    @Override
    public String toString()
    {
        return "<" + OpcodeName() + ">";
    }
    
    /**
     * Return true if the Xobject is a variable.
     */
    public boolean isVariable()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case VAR:
        case REG:
            return true;
        default:
            return false;
        }
    }

    public boolean isLocalOrParamVar()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case VAR:
            return isScopeLocal() || isScopeParam();
        case REG:
            return true;
        default:
            return false;
        }
    }

    public boolean isTempVar()
    {
        return code == Xcode.REG;
    }

    public boolean isVarAddr()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case VAR_ADDR:
            /* case Xcode.REG: */
            return true;
        default:
            return false;
        }
    }

    public boolean isLocalOrParamVarAddr()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case VAR_ADDR:
            return isScopeLocal() || isScopeParam();
        default:
            return false;
        }
    }

    public boolean isArray()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case VAR:
            return (isScopeParam() && type.isArray());
        case VAR_ADDR:
            return isScopeParam();
        default:
            return false;
        }
    }

    public boolean isArrayAddr()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case ARRAY_ADDR:
            return true;
        default:
            return false;
        }
    }

    /** return true if this is an constant object */
    public boolean isConstant()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case ARRAY_ADDR:
        case FUNC_ADDR:
        case VAR_ADDR:
        case INT_CONSTANT:
        case STRING_CONSTANT:
        case LONGLONG_CONSTANT:
        case FLOAT_CONSTANT:
        case MOE_CONSTANT:
        case SIZE_OF_EXPR:
        case GCC_ALIGN_OF_EXPR:
        case IDENT:
            return true;
        }
        return false;
    }
    
    public boolean isVarRef()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case VAR:
        case VAR_ADDR:
        case ARRAY_ADDR:
            return true;
        }
        
        return false;
    }

    /** return true if this object is zero integer constant. */
    public boolean isZeroConstant() {
      return false;
    }

    /** return true if this object is one integer constant. */
    public boolean isOneConstant() {
      return false;
    }

    /** return true if this object is integer constant */
    public boolean isIntConstant()
    {
        return code == Xcode.INT_CONSTANT;
    }

    /** return ture if this object is an assignment. */
    public boolean isSet()
    {
        return code == Xcode.ASSIGN_EXPR ||
            code == Xcode.F_ASSIGN_STATEMENT;
    }

    /** return true if this object is binary operation. */
    public boolean isBinaryOp()
    {
        return code.isBinaryOp();
    }

    /** return true if this object is unary operation. */
    public boolean isUnaryOp()
    {
        return code.isUnaryOp();
    }

    /** return ture if this object is an assignment with binary operation */
    public boolean isAsgOp()
    {
        return code.isAsgOp();
    }

    /** return true if this object is a terminal object */
    public boolean isTerminal()
    {
        return (this instanceof Ident) || code.isTerminal();
    }

    /**
     * return true if this object is a logical operation. EQ, NEQ, GE, GT, LE,
     * LT, AND, OR, NOT
     */
    public boolean isLogicalOp()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case LOG_EQ_EXPR:
        case LOG_NEQ_EXPR:
        case LOG_GE_EXPR:
        case LOG_GT_EXPR:
        case LOG_LE_EXPR:
        case LOG_LT_EXPR:
        case LOG_AND_EXPR:
        case LOG_OR_EXPR:
        case LOG_NOT_EXPR:
            return true;
        }
        return false;
    }
    
    public boolean isPragma()
    {
        if(code == null)
            return false;
        
        switch(code) {
        case PRAGMA_LINE:
        case OMP_PRAGMA:
        case XMP_PRAGMA:
            return true;
        }
        
        return false;
    }

    @Override
    public boolean enter(XobjectVisitor visitor)
    {
        return visitor.enter(this);
    }

    /* 
     * allocating new codes 
     */
    public static Xcode newCode() throws XmException
    {
        return Xcode.assign();
    }
    
    public boolean isExternalCode()
    {
        if(code == null)
            return false;
        return code.isAssignedCode();
    }
    
    public topdownXobjectIterator topdownIterator()
    {
        topdownXobjectIterator ite = new topdownXobjectIterator(this);
        ite.init();
        return ite;
    }

    public bottomupXobjectIterator bottomupIterator()
    {
        bottomupXobjectIterator ite = new bottomupXobjectIterator(this);
        ite.init();
        return ite;
    }
    
    public Ident getMember(String name)
    {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public IXobject getParent()
    {
        return parent;
    }
    
    @Override
    public void setParentRecursively(IXobject parent)
    {
        this.parent = parent;
    }

    @Override
    public Ident find(String name, int find_kind)
    {
        return null;
    }
    
    public Xobject cfold(Block block)
    {
      /* default */
      return this.copy();
    }

    public Xobject lbound(int dim)
    {
      Fshape shape = new Fshape(this);
      return shape.lbound(dim);
    }
    public Xobject lbound(int dim, Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.lbound(dim);
    }
  
    public Xobject[] lbounds()
    {
      Fshape shape = new Fshape(this);
      return shape.lbounds();
    }
    public Xobject[] lbounds(Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.lbounds();
    }

    public Xobject ubound(int dim)
    {
      Fshape shape = new Fshape(this);
      return shape.ubound(dim);
    }
    public Xobject ubound(int dim, Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.ubound(dim);
    }

    public Xobject[] ubounds()
    {
      Fshape shape = new Fshape(this);
      return shape.ubounds();
    }
    public Xobject[] ubounds(Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.ubounds();
    }

    public Xobject extent(int dim)
    {
      Fshape shape = new Fshape(this);
      return shape.extent(dim);
    }
    public Xobject extent(int dim, Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.extent(dim);
    }
    public Xobject[] extents()

    {
      Fshape shape = new Fshape(this);
      return shape.extents();
    }
    public Xobject[] extents(Block block)
    {
      Fshape shape = new Fshape(this, block);
      return shape.extents();
    }

    public Ident findVarIdent(String name)
    {
        return find(name, IXobject.FINDKIND_VAR);
    }
    public Ident findCommonIdent(String name)
    {
	return find(name, IXobject.FINDKIND_COMMON);
    }

    public boolean isEmptyList()
    {
	return (this instanceof XobjList && ((XobjList)this).Nargs() == 0);
    }

    public boolean hasNullArg()
    {
      if (this instanceof XobjList)
        return ((XobjList)this).hasNullArg();
      return false;
    }
}
