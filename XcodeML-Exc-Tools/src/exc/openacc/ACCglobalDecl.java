package exc.openacc;

import exc.block.Bcons;
import exc.block.BlockList;
import exc.object.*;

public class ACCglobalDecl{
  private final XobjectFile   _env;
  private final XobjList _globalConstructorFuncBody;
  private final XobjList _globalDestructorFuncBody;
  private final XobjectFile _env_device;
  private static final String ACC_INIT_FUNC_NAME = "_ACC_init";
  private static final String ACC_FINALIZE_FUNC_NAME = "_ACC_finalize";

  public ACCglobalDecl(XobjectFile env) {
    _env = env;
    _globalConstructorFuncBody = Xcons.List();
    _globalDestructorFuncBody = Xcons.List();
    _env_device = new XobjectFile();
    _env_device.setIdentList(Xcons.IDList());//_env.getGlobalIdentList().copy());
  }
  
  public XobjectFile getEnv() {
    return _env;
  }
  
  public XobjectFile getEnvDevice(){
    return _env_device;
  }

  public void setupGlobalConstructor() {
    Ident argv = Ident.Param("argv", Xtype.Pointer(Xtype.Pointer(Xtype.charType)));   // create "int argc" & "char **argv"
    Ident argc = Ident.Param("argc", Xtype.intType);
    XobjList params = Xcons.IDList();
    params.add(argc);
    params.add(argv);
    
    XobjList args = Xcons.List(argc.Ref(), argv.Ref());
    
    _globalConstructorFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc(ACC_INIT_FUNC_NAME).Call(args)));
    Xtype funcType = Xtype.Function(Xtype.voidType);

    Ident funcId = _env.declStaticIdent("acc_init_all", funcType);

    _env.add(XobjectDef.Func(funcId, params, null, Xcons.List(Xcode.COMPOUND_STATEMENT,
                                                            (Xobject)null, null, _globalConstructorFuncBody)));
  }

  public void setupGlobalDestructor() {
    _globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc(ACC_FINALIZE_FUNC_NAME).Call(null)));

    Xtype funcType = Xtype.Function(Xtype.voidType);
    Ident funcId = _env.declStaticIdent("acc_finalize_all", funcType);
    _env.add(XobjectDef.Func(funcId, null, null, Xcons.List(Xcode.COMPOUND_STATEMENT,
                             (Xobject)null, null, _globalDestructorFuncBody)));
  }
  
  public void setupMain(){
    topdownXobjectDefIterator ite = new topdownXobjectDefIterator(_env);
    for(ite.init(); !ite.end(); ite.next()){
      XobjectDef def = ite.getDef();
      if(def.isFuncDef() && def.getName().equals("main")){
        try{
          addArgsIntoMain(def);
          replaceMain(def);
        }catch(ACCexception e){
          ACC.fatal(e.getMessage());
        }
        break;
      }
    }
  }
  
  public void addGlobalConstructor(Xobject x){
    _globalConstructorFuncBody.add(x);
  }
  
  public void addGlobalDestructor(Xobject x){
    _globalDestructorFuncBody.add(x);
  }

  Ident declExternFunc(String funcName) {
    return ACCutil.getMacroFuncId(funcName, Xtype.voidType);
  }

  public Ident declExternIdent(String name, Xtype t) {
    return _env.declExternIdent(name, t);
  }
  public void finalize() {
    _env.collectAllTypes();
    _env.fixupTypeRef();
  }
  public Ident findVarIdent(String name){
    return _env.findVarIdent(name);
  }
  
  private void addArgsIntoMain(XobjectDef mainXobjDef) throws ACCexception {
    Xobject args = mainXobjDef.getFuncIdList();
    int numArgs = args.Nargs();
    Ident argc = Ident.Param("argc", Xtype.intType);
    Ident argv = Ident.Param("argv", Xtype.Pointer(Xtype.Pointer(Xtype.charType)));
    Ident funcId = findVarIdent("main");

    if(numArgs == 1){
      args.add(argv);
      ((FunctionType)funcId.Type()).setFuncParamIdList(args);
    }else if(numArgs == 0){
      args.add(argc);
      args.add(argv);
      ((FunctionType)funcId.Type()).setFuncParamIdList(args);
    }

    // Check arguments
    Xobject first_arg  = args.getArgOrNull(0);
    Xobject second_arg = args.getArgOrNull(1);

    checkFirstArg(first_arg);
    checkSecondArg(second_arg);
  }

  private void checkFirstArg(Xobject arg) throws ACCexception{
    if(!arg.Type().isBasic()){
      throw new ACCexception("Type of first argument in main() must be an interger.");
    }
    if(arg.Type().getBasicType() != BasicType.INT){
      throw new ACCexception("Type of first argument in main() must be an interger.");
    }
  }

  private void checkSecondArg(Xobject arg) throws ACCexception{
    if(!arg.Type().isPointer()){
      throw new ACCexception("Type of second argument in main() must be char **.");
    }

    boolean flag = false;
    if(arg.Type().getRef().isPointer() && arg.Type().getRef().getRef().isBasic()){
      if(arg.Type().getRef().getRef().getBasicType() == BasicType.CHAR){
        flag = true;
      }
    }

    if(!flag){
      throw new ACCexception("Type of second argument in main() must be char **.");
    }
  }
  
  private void replaceMain(XobjectDef mainXobjDef) {//throws ACCexception {
    Ident mainId = _env.findVarIdent("main");
    Xtype mainType = mainId.Type().getBaseRefType();

    XobjList mainIdList = (XobjList)mainXobjDef.getFuncIdList();
    Xobject mainDecls = mainXobjDef.getFuncDecls();
    Xobject mainBody = mainXobjDef.getFuncBody();

    Ident accMain = _env.declStaticIdent("acc_main", Xtype.Function(mainType));
    Ident accInitAll = _env.findVarIdent("acc_init_all");//_env.declExternIdent("acc_init_all", Xtype.Function(Xtype.voidType));
    Ident accFinalizeAll = _env.findVarIdent("acc_finalize_all");//_env.declExternIdent("acc_finalize_all", Xtype.Function(Xtype.voidType));

    _env.add(XobjectDef.Func(accMain, mainIdList, mainDecls, mainBody));

    BlockList newMainBody = Bcons.emptyBody();
    XobjList args = ACCutil.getRefs(mainIdList);
    newMainBody.add(accInitAll.Call(args));
    
    if(mainType.equals(Xtype.voidType)){
      newMainBody.add(accMain.Call(args));
      newMainBody.add(accFinalizeAll.Call(null));
    }else{
      Ident r = Ident.Local("r", mainType);
      newMainBody.addIdent(r);
      newMainBody.add(Xcons.Set(r.Ref(), accMain.Call(args)));
      newMainBody.add(accFinalizeAll.Call(null));
      newMainBody.add(Xcons.List(Xcode.RETURN_STATEMENT, r.Ref()));
    }

    XobjList newMain = Xcons.List(Xcode.FUNCTION_DEFINITION, mainId, mainIdList, mainDecls, newMainBody.toXobject());
    mainXobjDef.setDef(newMain);
  }
}
