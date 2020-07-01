package exc.xcalablemp;
import exc.block.*;
import exc.object.*;
import xcodeml.util.XmOption;

/**
 * XcalableMP AST translator
 */
public class XMPtranslate implements XobjectDefVisitor {
  private XMPglobalDecl			_globalDecl;
  private XMPtranslateGlobalPragma	_translateGlobalPragma;
  private XMPtranslateLocalPragma	_translateLocalPragma;
  private XMPrewriteExpr		_rewriteExpr;
  private boolean                       _all_profile;
  private boolean                       _selective_profile;

  public XMPtranslate(XMPglobalDecl globalDecl) {
    // FIXME current implementation only supports C language
    if (!XmOption.isLanguageC())
      XMP.fatal("current version only supports C language.");

    _globalDecl = globalDecl;
    _translateGlobalPragma = new XMPtranslateGlobalPragma(globalDecl);
    _translateLocalPragma = new XMPtranslateLocalPragma(globalDecl);
    _rewriteExpr = new XMPrewriteExpr(globalDecl);

    _all_profile = false;
    _selective_profile = false;
  }

  public void finish() {
    _globalDecl.finalizeGlobalDecl();
  }

  public void doDef(XobjectDef def) {
    translate(def);
  }

  private void translate(XobjectDef def) {
    if (!def.isFuncDef()) {
      Xobject x = def.getDef();
      switch (x.Opcode()) {
      case XMP_PRAGMA:
        _translateGlobalPragma.translate(x);
        break;

      case VAR_DECL:
        // for coarray declaration of XMP1.2
        _rewriteExpr.rewriteVarDecl(x, false);   // isLocal=false
        break;

      case ACC_PRAGMA:
        _rewriteExpr.rewriteACCPragma(def);
        break;
      }
      return;
    }
        
    FuncDefBlock fd = new FuncDefBlock(def);

    // fix subarrays
    fixSubArrayRef(fd);

    // translate directives
    _translateLocalPragma.translate(fd);

    // rewrite expressions
    _rewriteExpr.rewrite(fd);

    String funcName = fd.getBlock().getName();
    if(funcName == "main"){
      try{
	add_args_into_main(fd);   // ex) main() -> main(int argc, char **argv)
	create_new_main(fd);      // create new function "xmpc_main"
      }
      catch (XMPexception e) {
	Block b = fd.getBlock();
	XMP.error(b.getLineNo(), e.getMessage());
      }
    }
  }

  private void first_arg_check(Xobject arg) throws XMPexception{
    if(!arg.Type().isBasic()){
      throw new XMPexception("Type of first argument in main() must be an interger.");
    }
    if(arg.Type().getBasicType() != BasicType.INT){
      throw new XMPexception("Type of first argument in main() must be an interger.");
    }
  }

  private void second_arg_check(Xobject arg) throws XMPexception{
    if(!arg.Type().isPointer()){
      throw new XMPexception("Type of second argument in main() must be char **.");
    }

    boolean flag = false;
    if(arg.Type().getRef().isPointer() && arg.Type().getRef().getRef().isBasic()){
      if(arg.Type().getRef().getRef().getBasicType() == BasicType.CHAR){
	flag = true;
      }
    }

    if(!flag){
      throw new XMPexception("Type of second argument in main() must be char **.");
    }
  }
  
  // Create a new function xmpc_main() and copy main() to the new function
  private void create_new_main(FuncDefBlock fd) throws XMPexception {
    XobjectFile _env      = _globalDecl.getEnv();
    Ident mainId          = _globalDecl.findVarIdent("main");
    Xtype mainType        = ((FunctionType)mainId.Type()).getBaseRefType();

    if(!XmOption.getMainName().equals(""))
      mainId = _env.declGlobalIdent(XmOption.getMainName(), Xtype.Function(mainType));

    Xobject mainIdList    = fd.getDef().getFuncIdList();
    Xobject mainDecls     = fd.getDef().getFuncDecls();
    Xobject mainBody      = fd.getDef().getFuncBody();
    
    Ident xmpcInitAll     = _env.declExternIdent("xmp_init_all", Xtype.Function(Xtype.voidType));
    Ident xmpcMain        = _env.declGlobalIdent("xmpc_main", Xtype.Function(mainType));
    Ident xmpcFinalizeAll = _env.declExternIdent("xmp_finalize_all", Xtype.Function(Xtype.voidType));

    _env.add(XobjectDef.Func(xmpcMain, mainIdList, mainDecls, mainBody));

    XobjectDef mainXobjDef  = fd.getDef();
    BlockList newFuncBody   = Bcons.emptyBody();
    newFuncBody.setIdentList(Xcons.IDList());
    newFuncBody.setDecls(Xcons.List());
    Ident firstArg          = (Ident)fd.getDef().getFuncIdList().getArg(0);
    Ident secondArg         = (Ident)fd.getDef().getFuncIdList().getArg(1);
    XobjList mainIdAddrList = Xcons.List(firstArg, secondArg);
    
    newFuncBody.add(xmpcInitAll.Call(mainIdAddrList));
    if(mainType.equals(Xtype.voidType)){
      newFuncBody.add(xmpcMain.Call(mainIdList));
      newFuncBody.add(xmpcFinalizeAll.Call(null));
    }
    else{
      Ident r = Ident.Local("r", mainType);
      newFuncBody.addIdent(r);
      newFuncBody.add(Xcons.Set(r.Ref(), xmpcMain.Call(mainIdList)));
      newFuncBody.add(xmpcFinalizeAll.Call(null));
      newFuncBody.add(Xcons.List(Xcode.RETURN_STATEMENT, r.Ref()));
    }

    XobjList newfunc = Xcons.List(Xcode.FUNCTION_DEFINITION, mainId,
				  mainIdList, mainDecls, newFuncBody.toXobject());
    mainXobjDef.setDef(newfunc);
  }
  
  // This function is only used for main()
  // Add arguments of main() as belows
  //   main()                -> main(int argc, char **argv)
  //   main(int a)           -> main(int a, char **argv)
  //   main(int a, char **c) -> main(int a, char **c)   // no change
  // And check type of arguments
  private void add_args_into_main(FuncDefBlock fd) throws XMPexception {
    Xobject args = fd.getDef().getFuncIdList();
    int numArgs = args.Nargs();
    Ident argc = Ident.Param("argc", Xtype.intType);
    Ident argv = Ident.Param("argv", Xtype.Pointer(Xtype.Pointer(Xtype.charType)));
    Ident funcId = _globalDecl.findVarIdent("main");

    if(numArgs == 1){
      args.add(argv);
      ((FunctionType)funcId.Type()).setFuncParamIdList(args);
    }
    else if(numArgs == 0){
      args.add(argc);
      args.add(argv);
      ((FunctionType)funcId.Type()).setFuncParamIdList(args);
    }

    // Check arguments
    Xobject first_arg  = args.getArgOrNull(0);
    Xobject second_arg = args.getArgOrNull(1);

    first_arg_check(first_arg);
    second_arg_check(second_arg);
  }
  
  public void set_all_profile(){
    _all_profile = true;
    _translateLocalPragma.set_all_profile();
  }

  public void set_selective_profile(){
      _selective_profile = true;
      _translateLocalPragma.set_selective_profile();
  }

  public void setScalascaEnabled(boolean v) {
      _translateLocalPragma.setScalascaEnabled(v);
  }

  public void setTlogEnabled(boolean v) {
      _translateLocalPragma.setTlogEnabled(v);
  }

  private void fixSubArrayRef(FuncDefBlock def)
  {
    FunctionBlock fb = def.getBlock();
    if (fb == null) return;

    BlockIterator iter = new bottomupBlockIterator(fb);
    for (iter.init(); !iter.end(); iter.next()){
      Block block = iter.getBlock();

      XobjectIterator iter2 = new bottomupXobjectIterator(block.toXobject());
      for (iter2.init(); !iter2.end(); iter2.next()){
	Xobject x = iter2.getXobject();
	if (x != null && x.Opcode() == Xcode.SUB_ARRAY_REF){
          Boolean isStructure = (x.getArg(0).Opcode() == Xcode.MEMBER_ARRAY_REF);
	  String arrayName    = XMPutil.getArrayName(x);
	  Ident arrayId       = null;

	  if(! isStructure){
	    if (block.getBody() != null) arrayId = block.getBody().findLocalIdent(arrayName);
	    if (arrayId == null) arrayId = block.findVarIdent(arrayName);
	    if (arrayId == null) arrayId = _globalDecl.findVarIdent(arrayName);
	  }
	  else{
	    XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName);
	    arrayId = alignedArray.getArrayId();
	  }
	  if (arrayId == null) continue;

	  Xtype arrayType = arrayId.Type();
	  int n = arrayType.getNumDimensions();
	  XobjList subscripts = (XobjList)x.getArg(1);

	  for (int i = 0; i < n; i++, arrayType = arrayType.getRef()){
            if(subscripts.Nargs() == i){
              XMP.fatal(block.getLineNo(), "Invalid access of array");
            }
            else{
              long dimSize = arrayType.getArraySize();
              Xobject sizeExpr;
              if (dimSize == 0 || arrayType.getKind() == Xtype.POINTER){
                continue;
              }
              else if (dimSize == -1){
                sizeExpr = arrayType.getArraySizeExpr();
              }
              else {
                sizeExpr = Xcons.IntConstant((int)dimSize);
              }
              
              Xobject sub = subscripts.getArg(i);
              Xobject lb, len, st;
              
              if (! sub.isIndexRange()) continue;
              
              lb = ((XobjList)sub).getArg(0);
              if (lb == null) lb = Xcons.IntConstant(0);
              len = ((XobjList)sub).getArg(1);
              if (len == null) len = sizeExpr;
              st = ((XobjList)sub).getArg(2);
              if (st == null) st = Xcons.IntConstant(1);
              
              subscripts.setArg(i, Xcons.List(Xcode.INDEX_RANGE, lb, len, st));
            }
          }
	}
      }
    }
  }
}
