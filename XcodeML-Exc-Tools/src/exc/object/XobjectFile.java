/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import java.io.*;
import java.util.*;

/**
 * This object is a container for an object file represented in Xobject Format.
 * This contains: - type informations used in this file. - global symbol table -
 * information of global data - codes for functions
 */
public class XobjectFile extends XobjectDefEnv
{
    private List<Xtype> typeList = new ArrayList<Xtype>();
    protected HashMap<String, Xtype> typeIdTable = new HashMap<String, Xtype>();

    private String sourceFileName;
    private String compilerInfo;
    private String version;
    private String time;
    private String language;
    
    private List<String> headerLines;
    private List<String> tailerLines;
    
    private int gensym; // gensym counter

    public boolean debugFlag;
    public static boolean gcc_huge_common_bug;
    
  ///////private String tailText = "";     // for collectInit temporary version

    /** default constructor */
    public XobjectFile()
    {
    }

    static void fatal(String msg)
    {
        System.err.println("Fatal XobjectFile: " + msg);
        Thread.dumpStack();
        System.exit(1);
    }

    /** Return source file name */
    public String getSourceFileName()
    {
        return sourceFileName;
    }
    
    public String getCompilerInfo()
    {
        return compilerInfo;
    }
    
    public String getTime()
    {
        return time;
    }
    
    public String getVersion()
    {
        return version;
    }
    
    public String getLanguageAttribute()
    {
        return language;
    }

    public void setProgramAttributes(
        String sourceFileName, String language,
        String compilerInfo, String version, String time)
    {
        this.sourceFileName = sourceFileName;
        this.language = language;
        this.compilerInfo = compilerInfo;
        this.version = version;
        this.time = time;
    }
    
    //
    // Global Symbol table management
    // 

    private Ident createGlobalIdent(String name, Xtype t, StorageClass sclass)
    {
        Xtype addrType = t;
        Xcode addrCode;
        
        if(t.isArray()) {
            addrCode = Xcode.ARRAY_ADDR;
        } else if(t.isFunction()) {
            addrCode = Xcode.FUNC_ADDR;
        } else {
            addrCode = Xcode.VAR_ADDR;
        }
        
        addrType = Xtype.Pointer(t);
        
        return new Ident(name, sclass, t,
                         Xcons.Symbol(addrCode, addrType, name), VarScope.GLOBAL, null);
    }

    /**
     * Find the global identifer with name and type, and return Ident. If it is
     * not found, create global identifer as external (StorageClass:EXTDEF). If
     * found idetinfier mismatches given type, it cause fatal error.
     */
    public Ident declGlobalIdent(String name, Xtype t)
    {
        Ident id = findVarIdent(name);
        if(id != null) {
            if(id.Type().equals(t)) {
                switch(id.getStorageClass()) {
                case EXTERN:
                    id.setStorageClass(StorageClass.EXTDEF);
                case EXTDEF:
                case STATIC:
                    return id;
                }
            }
            fatal("declGlobalIdent: id is already defined," + id);
            return null;
        }
        
        id = createGlobalIdent(name, t, StorageClass.EXTDEF);
        identList.add(id);
        return id;
    }

    public Ident declStaticIdent(String name, Xtype t)
    {
        Ident id = findVarIdent(name);
        if(id != null) {
            if(id.Type().equals(t)) {
                switch(id.getStorageClass()) {
                case EXTERN:
                case EXTDEF:
                    id.setStorageClass(StorageClass.STATIC);
                case STATIC:
                    return id;
                }
            }
            fatal("declStaticIdent: id is already defined," + id);
            return null;
        }
        
        id = createGlobalIdent(name, t, StorageClass.STATIC);
        identList.add(id);
        return id;
    }
    
    public Ident declExternIdent(String name, Xtype t)
    {
        Ident id = findVarIdent(name);
        if(id != null) {
            if(id.Type().equals(t)) {
                switch(id.getStorageClass()) {
                case EXTERN:
                case EXTDEF:
                case STATIC:
                    return id;
                }
            }
            fatal("declExternIdent: id is already defined," + id);
            return null;
        }
        
        id = createGlobalIdent(name, t, StorageClass.EXTERN);
        identList.add(id);
        return id;
    }

    public String genExportSym(String leader, String parentFuncName)
    {
        String s = leader;
        s = s + "_" + parentFuncName + "_" + gensym++;
        s.intern();
        return s;
    }
    
    public String genSym(String leader)
    {
        String s = leader;
        s = s + "_" + gensym++;
        s.intern();
        return s;
    }

    public Xtype findType(String tid)
    {
        BasicType.TypeInfo ti = BasicType.getTypeInfoByName(tid);
        if(ti != null)
            return ti.type;
        
        Xtype t = typeIdTable.get(tid);
        if(t != null)
            return t;
        /* if not found create dummy type */
        t = new Xtype(Xtype.UNDEF, tid, 0, null);
        return t;
    }
    
    public void fixupTypeRef()
    {
        for(Iterator<Xtype> ite = typeList.iterator(); ite.hasNext(); ) {
            Xtype t = ite.next();
            if(t instanceof FunctionType) {
                FunctionType ft = (FunctionType)t;
                if(ft.ref.getKind() == Xtype.UNDEF)
                    ft.ref = findTypeId(ft.ref.Id());
                if(t.getFuncParam() != null) {
                    for(XobjArgs a = t.getFuncParam().getArgs(); a != null; a = a.nextArgs()) {
                        Xobject x = a.getArg();
                        if(x == null || x.type == null)
                            continue;
                        if(x.type.getKind() == Xtype.UNDEF)
                            x.type = findTypeId(x.type.Id());
                    }
                }
            } else if(t instanceof PointerType) {
                PointerType pt = (PointerType)t;
                if(pt.ref.getKind() == Xtype.UNDEF)
                    pt.ref = findTypeId(pt.ref.Id());
            } else if(t instanceof ArrayType) {
                ArrayType at = (ArrayType)t;
                if(at.ref.getKind() == Xtype.UNDEF)
                    at.ref = findTypeId(at.ref.Id());
                if(at.getArraySizeExpr() != null)
                    fixupTypeRefExpr(at.getArraySizeExpr());
            } else if(t instanceof CompositeType) {
                for(Xobject a : t.getMemberList()) {
                    Ident ident = (Ident)a;
                    if(ident.type.getKind() == Xtype.UNDEF) {
                        ident.type = findTypeId(ident.type.Id());
                    }
                }
            } else if(t instanceof FarrayType) {
                FarrayType at = (FarrayType)t;
                if(at.getRef().getKind() == Xtype.UNDEF)
                    at.setRef(findTypeId(at.getRef().Id()));
                if(at.getFarraySizeExpr() != null)
                    for(Xobject s : at.getFarraySizeExpr())
                        fixupTypeRefExpr(s);
            } else if(t.getClass().equals(Xtype.class)) {
                if(t.getKind() == Xtype.UNDEF) {
                    Xtype tt = findType(t.Id());
                    if(tt == null) {
                        throw new IllegalArgumentException("markType: undefined type ID=" + t.Id());
                    }
                    ite.remove();
                }
            }
        }
    }

    private void fixupTypeRefExpr(Xobject e)
    {
        if(e == null)
            return;
        XobjectIterator i = new topdownXobjectIterator(e);
        for(i.init(); !i.end(); i.next()) {
            Xobject x = i.getXobject();
            if(x != null && x.type != null && x.type.getKind() == Xtype.UNDEF)
                x.type = findTypeId(x.type.Id());
        }
    }

    Xtype findTypeId(String id)
    {
        Xtype t = typeIdTable.get(id);
        if(t != null)
            return t;
        fatal("findTypeId: type is not found, id=" + id);
        return null;
    }

    public void addHeaderLine(String s)
    {
        if(headerLines == null)
            headerLines = new ArrayList<String>();
        headerLines.add(s);
    }

    public List<String> getHeaderLines()
    {
        return headerLines;
    }
    
    public void addTailerLine(String s)
    {
        if(tailerLines == null)
            tailerLines = new ArrayList<String>();
        tailerLines.add(s);
    }
    
    public List<String> getTailerLines()
    {
        return tailerLines;
    }
    
    private void outputType(Xtype type, XobjectPrintWriter out)
    {
        out.print("{");
        out.print(Xtype.getKindName(type.getKind()));
        out.print(" ");
        out.printType(type);
        
        out.print(" (");
        
        if(type.isConst())          out.print(" const");
        if(type.isVolatile())       out.print(" volatile");
        if(type.isRestrict())       out.print(" restrict");
        if(type.isInline())         out.print(" inline");
        if(type.isArrayStatic())    out.print(" array_static");
        if(type.isFuncStatic())     out.print(" func_static");
        if(type.isFpublic())        out.print(" fpublic");
        if(type.isFprivate())       out.print(" fprivate");
        if(type.isFpointer())       out.print(" fpointer");
        if(type.isFoptional())      out.print(" foptional");
        if(type.isFtarget())        out.print(" ftarget");
        if(type.isFsave())          out.print(" fsave");
        if(type.isFparameter())     out.print(" fparameter");
        if(type.isFallocatable())   out.print(" fallocatable");
        if(type.isFintentIN())      out.print(" fintentIN");
        if(type.isFintentOUT())     out.print(" fintentOUT");
        if(type.isFintentINOUT())   out.print(" fintentINOUT");
        if(type.isFcrayPointer())   out.print(" fcrayPointer");  // (ID=060c)
        if(type.isFprogram())       out.print(" fprogram");
        if(type.isFintrinsic())     out.print(" fintrinsic");
        if(type.isFrecursive())     out.print(" frecursive");
        if(type.isFinternal())      out.print(" finternal");
        if(type.isFexternal())      out.print(" fexternal");
        if(type.isFsequence())      out.print(" fsequence");
        if(type.isFinternalPrivate()) out.print(" finternal_private");
        
        out.print(") ");

        if(type.copied != null) {
            out.printType(type.copied);
            out.println("}");
            return;
        }

        switch(type.getKind()) {
        case Xtype.BASIC: {
            BasicType.TypeInfo ti = BasicType.getTypeInfo(type.getBasicType());
            if(language.equalsIgnoreCase("f") || language.equalsIgnoreCase("fortran"))
                out.print(ti.fname);
            else
                out.print(ti.cname);
            break;
        }
        case Xtype.POINTER:
            out.printType(type.getRef());
            break;
        case Xtype.FUNCTION:
            out.printType(type.getRef());
            out.print(" ");
            out.printBool(type.isFuncProto());
            out.print(" ");
            out.printObjectNoIndent(type.getFuncParam());
            break;
        case Xtype.ARRAY:
            out.printType(type.getRef());
            out.print(" ");
            if(type.getArraySize() >= 0)
                out.printInt(type.getArraySize());
            else
                out.print("*");
            out.print(" ");
            if(type.getArraySizeExpr() != null)
                out.printObjectNoIndent(type.getArraySizeExpr());
            else
                out.print("*");
            break;
        case Xtype.ENUM:
        case Xtype.STRUCT:
        case Xtype.UNION:
            out.printIdentList(type.getMemberList(), 1);
            break;
        case Xtype.F_ARRAY:
            out.printType(type.getRef());
            if(type.getFarraySizeExpr() != null) {
                Xobject[] sizeExprs = type.getFarraySizeExpr();
                out.print(" (");
                for(Xobject sizeExpr : sizeExprs) {
                    out.print(" ");
                    out.printObject(sizeExpr);
                }
                out.print(")");
            }
            break;
        case Xtype.UNDEF:
            break;
        default:
            fatal("Output, bad type : " + type);
        }
        
        if(type.getGccAttributes() != null) {
            out.print(type.getGccAttributes());
        }
        
        if(type.getFkind() != null) {
            out.print(" ");
            out.print(type.getFkind());
        }
        
        out.println("}");
    }
    
    public void Output(Writer fout)
    {
        XobjectPrintWriter out = new XobjectPrintWriter(fout);

        /* dump types */
        for(Xtype type : typeList) {
            if(type.getKind() == Xtype.BASIC && !type.isConst() && !type.isVolatile())
                continue;
            outputType(type, out);
        }

        /* output global env */
        out.println("%");
        if(identList != null) {
            out.printIdentList(identList, 0);
        }

        /* output declaration and function */
        out.println("%");
        out.printDefs(getDefs());
        out.flush();
    }

    /* collect all used types */
    public void collectAllTypes()
    {
        typeList.clear();
        // collect types in GlobalIdentList
        collectType(identList);
        // collect types in GLobalDeclList
        topdownXobjectDefIterator ite = new topdownXobjectDefIterator(this);
        for(ite.init(); !ite.end(); ite.next()) {
            collectType(ite.getDef().getDef());
        }

        // clean up mark for the next phase
        for(Xtype type : typeList)
            type.is_marked = false;
    }
    
    public List<Xtype> getTypeList()
    {
        return typeList;
    }
    
    public void addType(Xtype type)
    {
        typeList.add(type);
        typeIdTable.put(type.Id(), type);
    }

    private void markType(Xtype t)
    {
        if(t == null || t.is_marked)
            return;
        
        t.is_marked = true;
        
        if(t.Id() == null) {
        	if(!t.isQualified()) {
        		if(t.isBasic())
        			return;
        		else if((t.isStruct() || t.isUnion() || t.isEnum()) &&
        			t.equals(t.getOriginal())) {
        			t.copied = t.getOriginal();
        			return;
        		}
        	}
        	
        	t.generateId();
        }
        
        // put on type list
        addType(t);

        switch(t.getKind()) {
        case Xtype.BASIC:
            collectType(t.getFkind());
            collectType(t.getFlen());
            break;
        case Xtype.POINTER:
            markType(t.getRef());
            break;
        case Xtype.ARRAY:
            markType(t.getRef());
            collectType(t.getArraySizeExpr());
            break;
        case Xtype.F_ARRAY:
            markType(t.getRef());
            for(Xobject x : t.getFarraySizeExpr())
                collectType(x);
            break;
        case Xtype.FUNCTION:
            markType(t.getRef());
            collectType(t.getFuncParam());
            break;
        case Xtype.STRUCT:
        case Xtype.UNION:
        case Xtype.ENUM:
            collectType(t.getMemberList());
            break;
        }
        
        collectType(t.getGccAttributes());

        if(t.copied != null)
            markType(t.copied);
        if(t.getOriginal() != null)
            markType(t.getOriginal());
    }

    private void collectType(Xobject x)
    {
        if(x == null)
            return;
        markType(x.Type());
        if(x instanceof XobjList) {
            for(Xobject a : (XobjList)x)
                collectType(a);
        } else if(x instanceof Ident) {
            Ident i = (Ident)x;
            collectType(i.getAddr());
            collectType(i.getEnumValue());
            collectType(i.getBitFieldExpr());
            collectType(i.getGccAttributes());
        }
    }


    /*
     *  handling Tail Text -- for collect init
     */
  /**************************
    public String getTailText() {
        return tailText;
    }
    public void clearTailText() {
        tailText = "";
    }
    public void addTailText(String text) {
        tailText += text;
    }
  **************************************/
}
