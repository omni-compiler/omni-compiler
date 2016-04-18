/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import java.util.LinkedList;
import java.util.List;

import exc.util.XobjectVisitable;
import exc.util.XobjectVisitor;
import xcodeml.ILineNo;
import xcodeml.IXobject;
import xcodeml.util.XmLog;
import xcodeml.util.XmOption;

/**
 * Definition in Xobject File iterator for definitions in Xobject file
 */
public class XobjectDef extends PropObject implements IXobject, XobjectVisitable, IVarContainer
{
  private XobjectDef parent;
  private XobjectDefEnv parent_env;
  private Xobject def;
  protected LinkedList<XobjectDef> child_defs = new LinkedList<XobjectDef>();
    
  public XobjectDef(Xobject def)
  {
    this(def, (XobjectDef)null);
  }
    
  public XobjectDef(Xobject def, XobjectDefEnv parent_env)
  {
    this(def, (XobjectDef)null);
    this.parent_env = parent_env;
  }
    
  public XobjectDef(Xobject def, XobjectDef parent)
  {
    this.def = def;
    this.parent = parent;
    
    switch(def.Opcode()) {
    case FUNCTION_DEFINITION: {
      Xobject body = def.getArg(3);
      if(body != null) {
	Xobject cont = body.getTail();
	if(cont != null && cont.Opcode() == Xcode.F_CONTAINS_STATEMENT) {
	  body.removeLastArgs();
	  for(Xobject d : (XobjList)cont) {
	    child_defs.add(new XobjectDef(d, this));
	  }
	}
      }
      break;
    }
    case F_MODULE_DEFINITION: {
      Xobject cont = def.getArg(3);
      if(cont != null && cont.Opcode() == Xcode.F_CONTAINS_STATEMENT) {
	def.setArg(3,null);  // remove body
	for(Xobject d : (XobjList)cont) {
	  child_defs.add(new XobjectDef(d, this));
	}
      }
      break;
    }
    }
  }
  
  // static constructor
  public static XobjectDef Func(Xobject name, Xobject id_list, Xobject decls, Xobject body)
  {
    return new XobjectDef(Xcons.List(Xcode.FUNCTION_DEFINITION,
				     name, id_list, decls, body));
  }

  public static XobjectDef Var(String name, Xobject initializer)
  {
    return new XobjectDef(Xcons.List(Xcode.VAR_DECL,
				     Xcons.Symbol(Xcode.IDENT, name), initializer));
  }
    
  public Xobject getDef()
  {
    return def;
  }
    
  public LinkedList<XobjectDef> getChildren()
  {
    return child_defs;
  }
    
  public boolean hasChildren()
  {
    return !child_defs.isEmpty();
  }

  public void setParent(XobjectDef parent)
  {
    this.parent = parent;
  }

  public void setParent(XobjectDefEnv env)
  {
    this.parent_env = env;
  }
    
  public XobjectDef getParent()
  {
    return parent;
  }
    
  public XobjectDefEnv getParentEnv()
  {
    XobjectDef pdef = parent;
    XobjectDefEnv penv = parent_env;
    while(penv == null) {
      if(pdef == null)
	throw new IllegalStateException("parent def is null");
      penv = pdef.parent_env;
      pdef = pdef.parent;
    }
        
    return penv;
  }

  public XobjectFile getFile()
  {
    XobjectDefEnv p = getParentEnv();
    return (XobjectFile)p;
  }

  public void setDef(Xobject def)
  {
    this.def = def;
  }
    
  public boolean isFuncDef()
  {
    return def != null && def.Opcode() == Xcode.FUNCTION_DEFINITION;
  }

  public boolean isFmoduleDef()
  {
    return def != null && def.Opcode() == Xcode.F_MODULE_DEFINITION;
  }

  public boolean isVarDecl()
  {
    return def != null && def.Opcode() == Xcode.VAR_DECL;
  }

  public boolean isFunctionDecl()
  {
    return def != null && def.Opcode() == Xcode.FUNCTION_DECL;
  }

  public boolean isBlockData()
  {
    return def != null && def.Opcode() == Xcode.F_BLOCK_DATA_DEFINITION;
  }

  public String getName()
  {
    Xobject nameObj = def.getArg(0);
    return nameObj != null ? nameObj.getSym() : null;
  }

  public Xobject getNameObj()
  {
    return def.getArg(0);
  }

  public void setName(String name)
  {
    def.setArg(0, Xcons.Symbol(Xcode.IDENT, name));
  }

  public Xtype getFuncType()
  {
    return def.getArg(0).Type();
  }

  public Xobject getInitializer()
  {
    return def.getArg(1);
  }

  public Xobject getFuncIdList()
  {
    return def.getArg(1);
  }

  public Xobject getFuncDecls()
  {
    return def.getArg(2);
  }

  public Xobject getFuncBody()
  {
    return def.getArgOrNull(3);
  }

  public Xobject getFuncGccAttributes()
  {
    return def.getArgOrNull(4);
  }
    
  public void insertBeforeThis(XobjectDef d)
  {
    if(parent != null)
      parent.child_defs.add(parent.child_defs.indexOf(this), d);
    else
      parent_env.insertBefore(this, d);
  }

  public void addAfterThis(XobjectDef d)
  {
    if(parent != null)
      parent.child_defs.add(parent.child_defs.indexOf(this) + 1, d);
    else
      parent_env.addAfter(this, d);
  }

  @Override
  public boolean enter(XobjectVisitor visitor)
  {
    return visitor.enter(this);
  }

  @Override
  public LineNo getLineNo()
  {
    return (def != null) ? def.getLineNo() : null;
  }
    
  public Ident findIdent(String name, int kind)
  {
    if(def.Opcode().isDefinition()) {
      XobjList identList = def.getIdentList();
      for(Xobject x : identList) {
	Ident id = (Ident)x;
	if(id.getName().equals(name)) {
	  switch(kind) {
	  case IXobject.FINDKIND_ANY:
	    return id;
	  case IXobject.FINDKIND_VAR:
	    if(id.getStorageClass().isVarOrFunc())
	      return id;
	    break;
	  case IXobject.FINDKIND_COMMON:
	    if(id.getStorageClass() == StorageClass.FCOMMON_NAME)
	      return id;
	    break;
	  case IXobject.FINDKIND_TAGNAME:
	    if(id.getStorageClass() == StorageClass.FTYPE_NAME ||
	       id.getStorageClass() == StorageClass.TAGNAME)
	      return id;
	    break;
	  }
	}
      }
    }
        
    if(parent != null) {
      return parent.findIdent(name, kind);
    } else if(parent_env != null) {
      return getFile().findIdent(name, kind);
    }
        
    return null;
  }
    
  public Ident findIdent(String name)
  {
    return findIdent(name, IXobject.FINDKIND_ANY);
  }

  public Ident findVarIdent(String name)
  {
    return findIdent(name, IXobject.FINDKIND_VAR);
  }

    public Ident findCommonIdent(String name)
    {
	return findIdent(name, IXobject.FINDKIND_COMMON);
    }   
 
  private Ident declFident(String name, Xtype t, boolean isFcommon)
  {
    if(def.Opcode() != Xcode.FUNCTION_DEFINITION) {
      XmLog.fatal("not function definition: " + def.OpcodeName());
      return null;
    }
        
    Ident id = findVarIdent(name);
    if(id != null) {
      if(id.Type().equals(t))
	return id;
      XmLog.fatal("id is already defined," + id);
      return null;
    }
        
    id = Ident.Fident(name, t, isFcommon, true, getFile());
    // do not add to ident_list now.
    // block will be moved possibly.
        
    return id;
  }

  public Ident declGlobalIdent(String name, Xtype t)
  {
    if(XmOption.isLanguageC())
      return getFile().declGlobalIdent(name, t);
        
    return declFident(name, t, true);
  }
    
  public Ident declExternIdent(String name, Xtype t)
  {
    if(XmOption.isLanguageC())
      return getFile().declExternIdent(name, t);
        
    return declFident(name, t, false);
  }

  public Ident declStaticIdent(String name, Xtype t)
  {
    if(XmOption.isLanguageC())
      return getFile().declStaticIdent(name, t);
        
    Ident id = declFident(name, t, true);
    id.setIsDelayedDecl(false);
    id.getAddr().setIsDelayedDecl(false);
    return id;
  }
    
  @Override
  public IXobject find(String name, int kind)
  {
    return findIdent(name, kind);
  }

  @Override
  public void setParentRecursively(IXobject parent)
  {
    if(getDef() != null)
      getDef().setParentRecursively(this);
  }
    
  @Override
  public String toString()
  {
    return "[def:" + ((def != null) ? def.OpcodeName() : null) + "]";
  }
}
