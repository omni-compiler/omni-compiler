package exc.xmpF;

import exc.object.*;
import exc.block.*;
import xcodeml.util.XmOption;

/**
 * XcalableMP AST translator:
 *  it implements XobjectDefVisitor, which applied to each XobjectDef 
 *  by the doDef method, using "visitor pattern".
 *  It contains the follwoing algorithm to comprise three phases.
 *   - XMPanalyzePragma: analysis XMP pragma and accumlate the information.
 *   - XMPrewriteExpr: rewrite expression according to the analysis.
 *   - XMPtransPragma: do the transformation for each XMP pragram.
 */
public class XMPtranslate implements XobjectDefVisitor
{
  BlockPrintWriter debug_out;
  XMPenv env;

  private final static String XMP_GENERATED_CHILD = "XMP_GENERATED_CHILD";

  // alorithms
  XMPanalyzePragma anaPragma = new XMPanalyzePragma();
  XMPrewriteExpr rewriteExpr = new XMPrewriteExpr();
  XMPtransPragma transPragma = new XMPtransPragma();
    
  static final String XMPmainFunc = "xmpf_main";

  /**
   * constructor without initial env.
   *   After the contruction, the object is to be initialized by "init"
   */
  public XMPtranslate() {  }
    
  /**
   * Constructor with initalized env.
   */
  public XMPtranslate(XobjectFile env) {
    init(env);
  }

  /**
   * Initialize the object with env.
   */
  public void init(XobjectFile env){
    this.env = new XMPenv(env);
    if(XMP.debugFlag)
      debug_out = new BlockPrintWriter(System.out);
  }
    
  /**
   * Finialize the env, that is, reflect all changes on block to Xobject.
   */
  public void finish() {
    env.finalize();
  }
    
  private void replace_main(XobjectDef d) {
    String name = d.getName();
    Ident id = env.getEnv().findVarIdent(name);
    if(id == null)
      XMP.fatal("'" + name + "' not in id_list");
    id.setName(XMPmainFunc);
    d.setName(XMPmainFunc);
  }

  static public void create_main(XobjectDef d){
    Ident mainId = Ident.FidentNotExternal("main", Xtype.FsubroutineType);
    BlockList newFuncBody = Bcons.emptyBody();

    newFuncBody.add(Ident.FidentNotExternal("xmpf_init_all_", Xtype.FsubroutineType).callSubroutine(null));
    newFuncBody.add(Ident.FidentNotExternal("xmpf_traverse_module", Xtype.FsubroutineType).callSubroutine(null));

    if(XmOption.isFonesided()){
      newFuncBody.add(Ident.FidentNotExternal("xmpf_traverse_countcoarray", Xtype.FsubroutineType).callSubroutine(null));
      newFuncBody.add(Ident.FidentNotExternal("xmpf_coarray_malloc_pool", Xtype.FsubroutineType).callSubroutine(null));
      newFuncBody.add(Ident.FidentNotExternal("xmpf_traverse_initcoarray", Xtype.FsubroutineType).callSubroutine(null));
      newFuncBody.add(Ident.FidentNotExternal("xmpf_sync_all_auto", Xtype.FsubroutineType).callSubroutine(null));
    }

    newFuncBody.add(Ident.FidentNotExternal("xmpf_main", Xtype.FsubroutineType).callSubroutine(null));
    newFuncBody.add(Ident.FidentNotExternal("xmpf_finalize_all_", Xtype.FsubroutineType).callSubroutine(null));

    XobjectDef newMain = XobjectDef.Func(mainId, null, null, newFuncBody.toXobject());
    newMain.getFuncType().setIsFprogram(true);
    d.addAfterThis(newMain);
  }
  
  private XobjectDef wrap_external(XobjectDef d){
    String name = d.getName();

    Xtype funcType = d.getFuncType().copy();
    //funcType.setFuncResultName(null);
    //Ident funcId = Ident.FidentNotExternal("xmpf_" + name, funcType);
    Ident funcId = Ident.FidentNotExternal("xmpf_" + name, Xtype.FsubroutineType);
    
    //((FunctionType)funcId.Type()).setFuncParam(funcType.getFuncParam());
    Xobject childParamList = Xcons.List(Xcode.ID_LIST);
    for (Xobject k: (XobjList)funcType.getFuncParam()){

      Xtype type  = ((XobjString)k).Type();
      if (type.getKind() == Xtype.F_ARRAY &&
	  !type.isFassumedShape() /*&&
				    XMParray.getArray(id) != null*/){
	childParamList.add(((XobjString)k).copy());
      }
    }
    ((FunctionType)funcId.Type()).setFuncParam(childParamList);

    funcId.setProp(XMP_GENERATED_CHILD, true);

    // generate child's ID list

    Xobject idList = d.getFuncIdList();
    Xobject childIdList = Xcons.List(Xcode.ID_LIST);
    int numAssumedId = 0;

    for (Xobject k: (XobjList)idList){
      Ident id = (Ident)k;
      Xtype type  = id.Type();
      if ((id.getStorageClass() == StorageClass.FPARAM &&
	   type.getKind() == Xtype.F_ARRAY &&
	   !type.isFassumedShape() /*&&
				     XMParray.getArray(id) != null*/)){
	childIdList.add(id.copy());
	numAssumedId++;
      }
      else if (id.getStorageClass() == StorageClass.FFUNC &&
	       !id.getName().equals(name)){
    	childIdList.add(id.copy());
      }
    }

    // generate child's declarations

    Xobject decls = d.getFuncDecls();
    Xobject childDecls = Xcons.List();
    int numAssumedDecl = 0;

    for (Xobject kk: (XobjList)decls){
      if (kk.getArg(0) == null) continue;
      if (kk.Opcode() == Xcode.F_COMMON_DECL ||
	  kk.Opcode() == Xcode.F_DATA_DECL ||
	  kk.Opcode() == Xcode.F_NAMELIST_DECL) continue;
      Ident id = d.findIdent(kk.getArg(0).getName());
      if (id == null) continue;
      Xtype type  = id.Type();

      if ((id.getStorageClass() == StorageClass.FPARAM &&
	   type.getKind() == Xtype.F_ARRAY &&
	   !type.isFassumedShape() /*&&
				     XMParray.getArray(id) != null*/)){
	childDecls.add(kk.copy());
	numAssumedDecl++;
      }
      else if (id.getStorageClass() == StorageClass.FFUNC &&
	       !id.getName().equals(name)){
	childDecls.add(kk.copy());
      }
    }

    if (numAssumedId == 0 && numAssumedDecl == 0) return null;
    
    // generate new body

    Xobject funcBody = d.getFuncBody();
    BlockList newFuncBody = Bcons.emptyBody();

    XobjectIterator i = new topdownXobjectIterator(funcBody);
    for (i.init(); !i.end(); i.next()) {
      Xobject xx = i.getXobject();
      if (xx != null){
	if (xx.Opcode() == Xcode.XMP_PRAGMA || xx.Opcode() == Xcode.OMP_PRAGMA){
	  String pragma = xx.getArg(0).getString();
	  if (pragma.equals("NODES") || pragma.equals("TEMPLATE") || pragma.equals("DISTRIBUTE") ||
	      pragma.equals("ALIGN") || pragma.equals("SHADOW") || pragma.equals("LOCAL_ALIAS") ||
	      pragma.equals("SAVE_DESC") ||
	      pragma.equals("COARRAY") || pragma.equals("THREADPRIVATE")){
	    Block pb = Bcons.PRAGMA(xx.Opcode(), xx.getArg(0).getString(), xx.getArg(1), null);
	    newFuncBody.add(pb);
	    i.setXobject(null);
	  }
	}
      }
    }

    // generate child

    XobjectDef newChild = XobjectDef.Func(funcId, childIdList, childDecls, funcBody);
    newChild.setParent(d);
    d.getChildren().add(newChild);

    // replace it

    Xobject args = Xcons.List();
    for (Xobject j: (XobjList)funcType.getFuncParam()){
      Ident id = d.findIdent(j.getName());
      Xtype type  = id.Type();
      if (type.getKind() != Xtype.F_ARRAY ||
	  type.isFassumedShape() /*||
				   XMParray.getArray(id) == null*/) continue;
      args.add(id.Ref());
    }

    newFuncBody.add(funcId.callSubroutine(args));
    // if (funcType.isFsubroutine())
    //   newFuncBody.add(funcId.callSubroutine(args));
    // else {
    //   Ident dummy = Ident.FidentNotExternal(XMP.genSym("XMP_dummy"), funcType.getRef());
    //   newFuncBody.add(Xcons.Set(dummy.Ref(), funcId.Call(args)));
    // }

    Xobject newDef = Xcons.List(Xcode.FUNCTION_DEFINITION, d.getNameObj(), (Xobject)idList,
				(Xobject)decls, newFuncBody.toXobject());
    d.setDef(newDef);

    return newChild;

  }

  public void copyXMParray(XobjectDef d){

    Xobject idList = d.getFuncIdList();
    for (Xobject k: (XobjList)idList){
      Ident id = (Ident)k;
      if (id.getStorageClass() == StorageClass.FPARAM){
	Ident idInParent = d.getParent().findIdent(id.getName());
	if (idInParent == null) XMP.fatal("internal error: not included in the parent function");
	XMParray array = XMParray.getArray(idInParent);

	if (array != null){

	  Xtype type = idInParent.Type();
	  int arrayDim = type.getNumDimensions();
	  Xobject sizeExprs[] = new Xobject[arrayDim];
	  for (int i = 0; i < arrayDim; i++){
	    Xobject lb;
	    if (array.isDistributed(i)){
	      sizeExprs[i] = Xcons.FindexRange(Xcons.IntConstant(0),
					       Xcons.binaryOp(Xcode.MINUS_EXPR,
							      array.getSizeVarAt(i),
							      Xcons.IntConstant(1)));
	    }
	    else {
	      sizeExprs[i] = Xcons.FindexRange(type.getFarraySizeExpr()[i].getArg(0),
					       type.getFarraySizeExpr()[i].getArg(1));
	    }
	  }
	  Xtype localType = Xtype.Farray(type.getRef(), sizeExprs);
	  localType.setTypeQualFlags(type.getTypeQualFlags());

	  String localName = XMP.PREFIX_ + id.getName();
	  Ident localId = Ident.FidentNotExternal(localName, localType);
	  localId.setStorageClass(id.getStorageClass());
	  localId.setValue(Xcons.Symbol(Xcode.VAR, localType, localName));

	  XMParray newArray = new XMParray(array, id, id.getName(), localId);
	  XMParray.setArray(id, newArray);
	}

      }
    }

  }


  // do transform takes three passes
  public void doDef(XobjectDef d) {
    FuncDefBlock fd = null;
    Boolean is_module = d.isFmoduleDef();
    XobjectDef newChild = null;

    XMP.resetError();

    if (isGeneratedProcedure(d))
      return;

    // System.out.println("def="+d.getDef());
    if(is_module){
      fd = XMPmoduleBlock(d);
      if(!haveXMPpragma(d.getDef()))
        return;
    } else if(d.isFuncDef()){ // declarations
      Xtype ft = d.getFuncType();
      if(ft != null && ft.isFprogram()) {
	ft.setIsFprogram(false);
	replace_main(d);
        create_main(d);
      }
      
      else if (d.getParent() == null){ // neither internal nor module procedures
      	newChild = wrap_external(d);
      }
      fd = new FuncDefBlock(d);
    } else if (d.isBlockData()){
      return;
    } else 
      XMP.fatal("Fotran: unknown decls");

    if(XMP.hasError())
      return;
        
    // distributed dummy arrays in the generated chidren shares their XMParray objects with
    // those in the parent.
    if (d.getNameObj().getProp(XMP_GENERATED_CHILD) != null){
      copyXMParray(d);
    }

    anaPragma.run(fd,env);
    if(XMP.hasError()) return;

    rewriteExpr.run(fd, env);
    if(XMP.hasError()) return;

    transPragma.run(fd,env);
    if(XMP.hasError()) return;

    // finally, replace body
    fd.Finalize();

    if(XMP.debugFlag) {
      System.out.println("**** final **** "+fd.getDef());
      XobjectPrintWriter out = new XobjectPrintWriter(System.out);
      out.print(fd.getDef().getDef());
      System.out.println("---- final ---- ");
    }

    if (newChild != null) doDef(newChild);

  }

  FuncDefBlock XMPmoduleBlock(XobjectDef def){
    XobjList decls = Xcons.List(); // emptyList
    Xobject xmp_pragma_list = Xcons.FstatementList();

    Xobject d = def.getDef();
    // module = (F_MODULE_DEFINITION name_id id_list decls body?)
    for(Xobject decl: (XobjList)d.getArg(2)){
      if(decl.Opcode() == Xcode.XMP_PRAGMA)
	xmp_pragma_list.add(decl);
      else
	decls.add(decl);
    }
    d.setArg(2,decls);
    d.setArg(3, xmp_pragma_list); // put xmp_pragmas as a body
    // System.out.println("module fblock="+d);
    return new FuncDefBlock(def);
  }

  boolean haveXMPpragma(Xobject x)
  {
    XobjectIterator i = new topdownXobjectIterator(x);
    for(i.init(); !i.end(); i.next()) {
      Xobject xx = i.getXobject();
      if(xx != null && xx.Opcode() == Xcode.XMP_PRAGMA)
	return true;
    }
    return false;
  }

  private boolean isGeneratedProcedure(XobjectDef d) {
    String name = d.getName();
    if (name != null && name.startsWith("xmpf_traverse_"))
      return true;

    return false;
  }
}
