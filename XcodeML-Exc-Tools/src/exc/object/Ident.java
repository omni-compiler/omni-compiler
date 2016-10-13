package exc.object;
import exc.block.*;

import java.util.ArrayList;
import java.util.List;

import exc.util.XobjectVisitor;
import xcodeml.util.XmLog;
import xcodeml.util.XmOption;

/**
 * Represents identifier
 */
public class Ident extends Xobject
{
    /** stroage class */
    private StorageClass stg_class;
    /** key */
    private String name;
    /** base address expression value */
    private Xobject value;
    /** declared in (VAR_DECL ) */
    private boolean declared;
    /** for TEMP_VAR */
    private int num;
    /** bit-field */
    private int bit_field;
    /** bit-field expression */
    private Xobject bit_field_expr;
    /** enum member value for MOE */
    private Xobject enum_value;
    /** C: gcc attributes */
    private Xobject gcc_attrs;
    /** Fortran: common vars for common name id */
    private List<Ident> fcommon_vars;
    /** Fortran: common name for common vars */
    private String fcommon_name;
    /** Fortran: parameter's initial value */
    private Xobject fparam_value;
    /** Fortran: declared module */
    private String declared_module;
    /** Codimensions for coarray (#284) */
    private Xobject codimensions;      // Codimensions might be moved into this.type like Fortran.
                                       // See exc.object.FarrayType
  
    // constructor
    public Ident(String name, StorageClass stg_class, Xtype type, Xobject v,
                 VarScope scope)
    {
        this(name, stg_class, type, v, 0, null, 0, null, null, null, null);
        setScope(scope);
    }
    public Ident(String name, StorageClass stg_class, Xtype type, Xobject v,
                 VarScope scope, Xobject codimensions)
    {
        this(name, stg_class, type, v, 0, null, 0, null, null, null, codimensions);
        setScope(scope);
    }
    
    // constructor
    public Ident(String name, StorageClass stg_class, Xtype type, Xobject v,
                 int optionalFlags, Xobject gccAttrs,
                 int bit_field, Xobject bit_field_expr, Xobject enum_value,
                 Xobject fparam_value, Xobject codimensions)
    {
        super(null, type, optionalFlags);
        if(name != null)
            this.name = name.intern();
        this.stg_class = stg_class;
        this.value = v;
        this.declared = false;
        this.bit_field = bit_field;
        this.bit_field_expr = bit_field_expr;
        this.gcc_attrs = gccAttrs;
        this.enum_value = enum_value;
        this.fparam_value = fparam_value;
        this.codimensions = codimensions;
    }
  /************************
    // for upper-compatibility
    public Ident(String name, StorageClass stg_class, Xtype type, Xobject v,
                 int optionalFlags, Xobject gccAttrs,
                 int bit_field, Xobject bit_field_expr, Xobject enum_value,
                 Xobject fparam_value) {
      this(name, stg_class, type, v,
           optionalFlags, gccAttrs,
           bit_field, bit_field_expr, enum_value,
           fparam_value, null);
    }
  ****************************/

    // constructor
    public Ident(String name, StorageClass stg_class, Xtype type, Xobject v,
                 boolean declared, int optionalFlags, Xobject gccAttrs,
                 int bit_field, Xobject bit_field_expr, Xobject enum_value,
                 Xobject fparam_value, Xobject codimensions) {
      super(null, type, optionalFlags);
      if(name != null) {
        this.name = name.intern();
      }

      this.stg_class = stg_class;
      this.value = v;
      this.declared = declared;
      this.bit_field = bit_field;
      this.bit_field_expr = bit_field_expr;
      this.gcc_attrs = gccAttrs;
      this.enum_value = enum_value;
      this.fparam_value = fparam_value;
      this.codimensions = codimensions;
    }
  /***************************************:
    // for upper-compatibility
    public Ident(String name, StorageClass stg_class, Xtype type, Xobject v,
                 boolean declared, int optionalFlags, Xobject gccAttrs,
                 int bit_field, Xobject bit_field_expr, Xobject enum_value,
                 Xobject fparam_value) {
      this(name, stg_class, type, v,
           declared, optionalFlags, gccAttrs,
           bit_field, bit_field_expr, enum_value,
           fparam_value, null);
    }
  **********************************/

    // constructor for temporary variable
    public Ident(int num, Xtype type)
    {
        this(null, StorageClass.REG, type, null, null, null);
        this.num = num;
    }
    
    public StorageClass getStorageClass()
    {
        return stg_class;
    }
    
    public void setStorageClass(StorageClass stg_class)
    {
        this.stg_class = stg_class;
    }

    @Override
    public String getName()
    {
        if(stg_class == StorageClass.REG)
            return "r_" + Integer.toHexString(num);
        else
            return name;
    }

    public int getFrank()
    {
        return type.getNumDimensions();
    }

    public void setName(String name)
    {
        this.name = name;
    }

    @Override
    public String getSym()
    {
        return getName();
    }
    
    @Override
    public String getString()
    {
        return getName();
    }
    
    public Xobject getAddr()
    {
        // address of array argument is the base address of array, not variable.
        if(stg_class == StorageClass.PARAM && type.isArray())
            return Ref();
        return value;
    }

    public Xobject getValue()
    {
        return value;
    }

   public void setValue(Xobject value)
    {
        this.value = value;
    }

   public Xobject getCodimensions()     // for coarray C (#284)
    {
        return codimensions;
    }

   public void setCodimensions(Xobject codimensions)     // for coarray C (#284)
    {
        this.codimensions = codimensions;
    }

    public int getCorank()
    {
      if (codimensions != null) {                // for coarray C temporarily (#284)
        return codimensions.Nargs() + 1;
      } else {                                   // for coarray Fortran (#060)
        return (Type() == null) ? 0 : Type().getCorank();
      }
    }

    public boolean isCoarray()          // commonly for C (#284) and Fortran (#060)
    {
        return (getCorank() > 0);
    }

    public boolean wasCoarray()           // for coarray Fortran (#060)
    {
        return (Type() == null) ? false : Type().wasCoarray();
    }


    public boolean isDeclared()
    {
        return declared;
    }

    /** @deprecated setIsDeclared() */
    public void Declared()
    {
        declared = true;
    }
    
    public void setIsDeclared(boolean declared)
    {
        this.declared = declared;
    }

    public int regn()
    {
        return num;
    }

    public void setBitField(int n)
    {
        bit_field = n;
    }

    public int getBitField()
    {
        return bit_field;
    }
    
    public Xobject getBitFieldExpr()
    {
        return bit_field_expr;
    }
    
    public Xobject getEnumValue()
    {
        return enum_value;
    }
    
    public Xobject getGccAttributes()
    {
        return gcc_attrs;
    }
    
    public final boolean isUsedInArraySize()
    {
        return getOptionalFlag(OPT_USED_IN_ARRAY_SIZE);
    }

    public final void setIsUsedInArraySize(boolean enabled)
    {
        setOptionalFlag(OPT_USED_IN_ARRAY_SIZE, enabled);
    }
    
    @Override
    public Xobject cfold(Block block)
    {
      if (Type().isFparameter() && fparam_value != null) {
        // I don't know why but fparam_value is always in this form.
        if (fparam_value.Nargs() == 2 && fparam_value.getArg(1) == null) {
          Xobject value = fparam_value.getArg(0);
          return value.cfold(block);
        } else {
          XmLog.fatal("Ident.cfold: unknown form of fparam_value");
        }
      }

      if (declared_module != null) {
        XobjectDefEnv xobjDefEnv = ((FunctionBlock)block).getEnv();
        XobjectFile xobjFile = (XobjectFile)xobjDefEnv;

        if (xobjFile.findVarIdent(declared_module) == null)
          XmLog.fatal("Ident.cfold: not found module name in globalSymbols: " + 
                      declared_module);

        for (XobjectDef punit: xobjFile.getDefs()) {
          if (declared_module.equals(punit.getName())) {
            // found the module that declares this ident
            Ident ident2 = punit.getDef().findVarIdent(name);
            return ident2.cfold(block);
          }
        }
      }

      return this.copy();
    }

    @Override
    public Xobject copy() {
      return new Ident(name, stg_class, type, value, declared, getOptionalFlags(),
                       gcc_attrs, bit_field, bit_field_expr, enum_value,
                       fparam_value, codimensions);
    }

    @Override
    public boolean equals(Xobject x)
    {
        /* note: id is not replicated. */
        return this == x;
    }

    @Override
    public String toString()
    {
        if(getStorageClass() == StorageClass.REG) {
            return "[0x" + Integer.toHexString(num) + " " + (type == null ? "*" : type.toString())
                + " () *]";
        }
        
        StringBuilder b = new StringBuilder(256);
        b.append("[");
        b.append(name == null ? "*" : name);
        b.append(" ");
        b.append(stg_class == null ? "*" : stg_class.toXcodeString());
        b.append(" ");
        b.append(type == null ? "*" : type.toString());
        b.append(" ");
        b.append(value == null ? "()" : value.toString());

        /* bit_field, bit_field_expr, enum_value are exclusively set */
        b.append(" ");
        if(bit_field != 0 || bit_field_expr != null) {
            if(bit_field != 0)
                b.append(bit_field);
            else
                b.append(bit_field_expr.toString());
        } else if(enum_value != null) {
            b.append(" ");
            b.append(enum_value.toString());
        } else {
            b.append("*");
        }

        b.append(declared ? " D" : "");
        
        if(gcc_attrs != null) {
            b.append(" ");
            b.append(gcc_attrs.toString());
        }
        
        b.append("]");
        
        return b.toString();
    }

    // return value for refernce 'id'
    public Xobject Ref()
    {
        if(getStorageClass() == StorageClass.REG)
            return Xcons.Int(Xcode.REG, type, num);
        if(type.isFunction())
            return value;
        if(value == null) {
            XmLog.fatal("value is null : " + toString());
        }
        return Xcons.PointerRef(value);
    }

    // return value for referenece 'id[i]' == *(id + i)
    public Xobject Index(int i)
    {
        return Index(Xcons.IntConstant(i));
    }

    public Xobject Index(Xobject i)
    {
        if(type.isArray() || value.code == Xcode.VAR_ADDR || value.code == Xcode.ARRAY_ADDR)
            return Xcons.PointerRef(Xcons.binaryOp(Xcode.PLUS_EXPR, Ref(), i));
        if(type.isPointer())
            return Xcons.PointerRef(Xcons.binaryOp(Xcode.PLUS_EXPR, value, i));
        XmLog.fatal("Index: not Pointer");
        return null;
    }

    public Xobject Call()
    {
        return Call(Xcons.List());
    }
    
    public Xobject Call(Xobject args)
    {
        if(type.isFunction())
            return Xcons.List(Xcode.FUNCTION_CALL, type.getRef(), value, args);
        else
            XmLog.fatal("Call: not Function");
        return null;
    }

    @Override
    public boolean enter(XobjectVisitor visitor)
    {
        if(bit_field_expr != null) {
            if(!visitor.enter(bit_field_expr))
                return false;
        }
        if(enum_value != null) {
            if(!visitor.enter(enum_value))
                return false;
        }
        return true;
    }
    
    //
    // static member to construct variable
    //
    public static Ident Var(String name, Xtype t, Xtype addrt, VarScope scope)
    {
      return Var(name, t, addrt, scope, null);
    }
    public static Ident Var(String name, Xtype t, Xtype addrt, VarScope scope,
                            Xobject codimensions)
    {
        StorageClass sclass;
        Xcode addrCode;
        
        if(XmOption.isLanguageC()) {
            sclass = (scope == VarScope.PARAM) ? StorageClass.PARAM : StorageClass.AUTO;
            addrCode = t.isArray() ? Xcode.ARRAY_ADDR : Xcode.VAR_ADDR;
        } else {
            sclass = (scope == VarScope.PARAM) ? StorageClass.FPARAM : StorageClass.FLOCAL;
            addrCode = Xcode.VAR;
        }
        
        return new Ident(
            name, sclass, t,
            Xcons.Symbol(addrCode, addrt, name), scope, codimensions);
    }
    
    public static Ident Local(String name, Xtype t, Xtype addrt)
    {
        if(XmOption.isLanguageC())
            return Var(name, t, addrt, VarScope.LOCAL);
        return Fident(name, t, false, null);
    }

    public static Ident Local(String name, Xtype t)
    {
        if(XmOption.isLanguageC())
            return Local(name, t, Xtype.Pointer(t));
        return Fident(name, t);
    }

    public static Ident Param(String name, Xtype t)
    {
        return Var(name, t, Xtype.Pointer(t), VarScope.PARAM);
    }
    
    public static Ident TempVar(int num, Xtype t)
    {
        if(t.isArray()) {
            t = Xtype.Pointer(t.getRef()); // convert to pointer
            return new Ident(num, t);
        } else
            return new Ident(num, t);
    }

    public static Ident Fident(String name, Xtype t)
    {
        return Fident(name, t, null);
    }
    
    public static Ident Fident(String name, Xtype t, XobjectFile xobjFile)
    {
        return Fident(name, t, false, xobjFile);
    }
    
    public static Ident Fident(String name, Xtype t, boolean isFcommon, XobjectFile xobjFile)
    {
        return Fident(name, t, isFcommon, true, xobjFile);
    }
    
    public static Ident FidentNotExternal(String name, Xtype t)
    {
        return Fident(name, t, false, false, null);
    }

    /**
     * Fortran: create identifier.
     * @param name
     *      symbol name
     * @param t
     *      type
     * @param isFcommon
     *      if is declare as common variable
     * @param isDecl
     *      1. copy type if type is function/subroutine which is not external if true.
     *      2. add declaration if true.
     * @param xobjFile
     *      XobjectFile in current context
     * @return
     *      created identifier
     */
    public static Ident Fident(String name, Xtype t, boolean isFcommon, boolean isDecl, XobjectFile xobjFile)
    {
        Xcode code;
        StorageClass sclass;

        if(t == null) {
            code = null;
            sclass = StorageClass.FCOMMON_NAME;
        } else {
            if(t.isFunction()) {
                code = Xcode.FUNC_ADDR;
                sclass = StorageClass.FFUNC;
                if(!t.isFexternal()) {
                    if(isDecl) {
                        if(t.isFsubroutine())
                            t = Xtype.FexternalSubroutineType.copy();
                        else if(t.getRef().isBool())
                            t = Xtype.FexternalLogicalFunctionType.copy();
                        else if(t.getRef().isIntegral())
                            t = Xtype.FexternalIntFunctionType.copy();
                        else {
                            if(xobjFile == null)
                                throw new IllegalStateException("xobjFile is null");
                            t = t.copy();
                            t.setIsFexternal(true);
                            t.generateId();
                            xobjFile.addType(t);
                        }
                    } else {
                        t = t.copy();
                    }
                }
            } else {
                code = Xcode.VAR;
                sclass = isFcommon ? StorageClass.FCOMMON : StorageClass.FLOCAL;
            }
        }
        
        Xobject addr = null;
        if(t == null || t.isFunction())
            isFcommon = false;
        
        if(code != null) {
            addr = Xcons.Symbol(code, t, name);
            addr.setIsDelayedDecl(isDecl);
            addr.setIsToBeFcommon(isFcommon);
        }
        
        Ident id = new Ident(name, sclass, t, addr, VarScope.LOCAL, null);
        id.setIsToBeFcommon(isFcommon);
        
        return id;
    }
    
    public void addFcommonVar(Ident id)
    {
        if(fcommon_vars == null)
            fcommon_vars = new ArrayList<Ident>();
        fcommon_vars.add(id);
    }
    
    public List<Ident> getFcommonVars()
    {
        return fcommon_vars;
    }
    
    public String getFcommonName()
    {
        return fcommon_name;
    }
    
    public void setFcommonName(String name)
    {
        fcommon_name = name;
    }
    
    public Xobject getFparamValue()
    {
        return fparam_value;
    }
    
    public void setFparamValue(Xobject value)
    {
        this.fparam_value = value;
    }

    public String getFdeclaredModule()
    {
        return declared_module;
    }
    
    public void setFdeclaredModule(String module_name)
    {
        this.declared_module = module_name;
    }

    // for Fortran subroutine
    public Xobject callSubroutine()
    {
      return Xcons.List(Xcode.EXPR_STATEMENT,Call());
    }

    public Xobject callSubroutine(Xobject args)
    {
      return Xcons.List(Xcode.EXPR_STATEMENT,Call(args));
    }
}
