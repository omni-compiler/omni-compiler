package exc.openacc;

import exc.block.Bcons;
import exc.block.BlockList;
import exc.block.FuncDefBlock;
import exc.object.*;
import exc.xcalablemp.XMPexception;

import java.util.*;

public class ACCglobalDecl{
  private static final String ACC_DESTRUCTOR_FUNC_PREFIX = "acc_traverse_finalize_file_";
  private static final String ACC_CONSTRUCTOR_FUNC_PREFIX = "acc_traverse_init_file_";
  private static final String ACC_TRAVERSE_INIT_FUNC_NAME = "acc_traverse_init";
  private static final String ACC_TRAVERSE_FINALIZE_FUNC_NAME = "acc_traverse_finalize";
  private XobjectFile   _env;
  private Map<String, FuncInfo> funcInfoMap;
  private XobjList _globalConstructorFuncBody;
  private XobjList _globalDestructorFuncBody;
  private XobjectFile _env_device;
  
  
  private static String ACC_INIT_FUNC_NAME = "_ACC_init";
  private static String ACC_FINALIZE_FUNC_NAME = "_ACC_finalize";
  private static String ACC_GPU_INIT_FUNC_NAME = "_ACC_gpu_init";
  private static String ACC_GPU_FINALIZE_FUNC_NAME = "_ACC_gpu_finalize";
  
  public ACCglobalDecl(XobjectFile env) {
    _env = env;
    funcInfoMap = new HashMap<String, FuncInfo>();
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
  
  public void setCalleeInfo(String funcName, int paramNum, Xcode type, boolean mustBeAllocated){
    FuncInfo funcInfo = funcInfoMap.get(funcName);
    if(funcInfo == null){
      funcInfo = new FuncInfo(funcName);
      funcInfoMap.put(funcName, funcInfo);
    }
    funcInfo.setParamInfo(paramNum, type, mustBeAllocated);
  }
  public void setCallerInfo(String funcName, int argNum, Xcode type, boolean isAllocated){
    FuncInfo funcInfo = funcInfoMap.get(funcName);
    if(funcInfo == null){
      funcInfo = new FuncInfo(funcName);
      funcInfoMap.put(funcName, funcInfo);
    }
    funcInfo.setArgInfo(argNum, type, isAllocated);
  }
  public void checkPresentData(){
    for(FuncInfo funcInfo : funcInfoMap.values()){
      for(FuncParam funcParam : funcInfo.funcParams.values()){
        System.out.println("check:func=" + funcInfo.funcName + ", arg=" + funcParam.paramNum);
        if(funcParam.mustBeAllocated){
          FuncArg funcArg = funcInfo.funcArgs.get(funcParam.paramNum);
          if(funcArg == null){
            ACC.fatal("func=" + funcInfo.funcName + ", arg=" + funcParam.paramNum + " is not allocated");
          }else{
            if(funcParam.type != funcArg.type){
              ACC.fatal("param type not equals to arg type");
            }
            if(! funcArg.isAllocated){
              ACC.fatal("func=" + funcInfo.funcName + ", arg=" + funcParam.paramNum + " is not allocated");
            }
          }
        }
      }
    }
  }
  public void setupGlobalConstructor() {
    //_globalConstructorFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc(ACC_GPU_INIT_FUNC_NAME).Call(null)));

    Ident argv = Ident.Param("argv", Xtype.Pointer(Xtype.Pointer(Xtype.charType)));   // create "int argc" & "char **argv"
    Ident argc = Ident.Param("argc", Xtype.intType);
    XobjList params = Xcons.IDList();
    params.add(argc);
    params.add(argv);
    
    XobjList args = Xcons.List(argc.Ref(), argv.Ref());
    
    //_globalConstructorFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc(ACC_INIT_FUNC_NAME).Call(args)));
    Xtype funcType = Xtype.Function(Xtype.voidType);

//    funcType.setGccAttributes(Xcons.List(Xcode.GCC_ATTRIBUTES,
//                                         Xcons.List(Xcode.GCC_ATTRIBUTE,
//                                                    new Ident("constructor", null, null, null, null),
//                                                    Xcons.List())));
    String fileName = getSourceBaseName();
    
    Ident funcId = _env.declExternIdent(ACC_CONSTRUCTOR_FUNC_PREFIX + fileName, funcType);

    _env.add(XobjectDef.Func(funcId, null, null, Xcons.List(Xcode.COMPOUND_STATEMENT,
                                                            (Xobject)null, null, _globalConstructorFuncBody)));
  }

  public void setupGlobalDestructor() {
    //_globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc(ACC_GPU_FINALIZE_FUNC_NAME).Call(null)));

    //_globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc(ACC_FINALIZE_FUNC_NAME).Call(null)));

    Xtype funcType = Xtype.Function(Xtype.voidType);
//    funcType.setGccAttributes(Xcons.List(Xcode.GCC_ATTRIBUTES,
//                                         Xcons.List(Xcode.GCC_ATTRIBUTE,
//                                                    new Ident("destructor", null, null, null, null),
//                                                    Xcons.List())));
    String fileName = getSourceBaseName();
    Ident funcId = _env.declExternIdent(ACC_DESTRUCTOR_FUNC_PREFIX + fileName, funcType);
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
          //renameMain(def);
          //createMain();
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

  public Ident declExternFunc(String funcName) {
    return ACCutil.getMacroFuncId(funcName, Xtype.voidType);
  }
  public String genSym(String prefix) {
    return _env.genSym(prefix);
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
    Xtype mainType = ((FunctionType)mainId.Type()).getBaseRefType();

    XobjList mainIdList = (XobjList)mainXobjDef.getFuncIdList();
    Xobject mainDecls = mainXobjDef.getFuncDecls();
    Xobject mainBody = mainXobjDef.getFuncBody();

    Ident accMain = _env.declStaticIdent("acc_main", Xtype.Function(mainType));
    //Ident accInitAll = _env.findVarIdent("acc_init_all");//_env.declExternIdent("acc_init_all", Xtype.Function(Xtype.voidType));
    //Ident accFinalizeAll = _env.findVarIdent("acc_finalize_all");//_env.declExternIdent("acc_finalize_all", Xtype.Function(Xtype.voidType));
    Ident accInit = declExternIdent(ACC_INIT_FUNC_NAME, Xtype.Function(Xtype.voidType));
    Ident accFinalize = declExternIdent(ACC_FINALIZE_FUNC_NAME, Xtype.Function(Xtype.voidType));
    Ident accTraverseInit = declExternIdent(ACC_TRAVERSE_INIT_FUNC_NAME, Xtype.Function(Xtype.voidType));
    Ident accTraverseFinalize = declExternIdent(ACC_TRAVERSE_FINALIZE_FUNC_NAME, Xtype.Function(Xtype.voidType));

    _env.add(XobjectDef.Func(accMain, mainIdList, mainDecls, mainBody));

    BlockList newMainBody = Bcons.emptyBody();
    XobjList args = ACCutil.getRefs(mainIdList);
    newMainBody.add(accInit.Call(args));
    newMainBody.add(accTraverseInit.Call());
    
    if(mainType.equals(Xtype.voidType)){
      newMainBody.add(accMain.Call(args));
      newMainBody.add(accTraverseFinalize.Call());
      newMainBody.add(accFinalize.Call(null));
    }else{
      Ident r = Ident.Local("r", mainType);
      newMainBody.addIdent(r);
      newMainBody.add(Xcons.Set(r.Ref(), accMain.Call(args)));
      newMainBody.add(accTraverseFinalize.Call());
      newMainBody.add(accFinalize.Call(null));
      newMainBody.add(Xcons.List(Xcode.RETURN_STATEMENT, r.Ref()));
    }

    XobjList newMain = Xcons.List(Xcode.FUNCTION_DEFINITION, mainId, mainIdList, mainDecls, newMainBody.toXobject());
    mainXobjDef.setDef(newMain);
  }
  
  
  
  private void renameMain(XobjectDef mainDef) {//throws ACCexception {
    Ident mainId = _env.findVarIdent("main");
    Xtype mainType = ((FunctionType)mainId.Type()).getBaseRefType();
    Xobject mainIdList = mainDef.getFuncIdList();
    Xobject mainDecls = mainDef.getFuncDecls();
    Xobject mainBody = mainDef.getFuncBody();
    Ident accMain = _env.declStaticIdent("_ACC_main", Xtype.Function(mainType));
    mainDef.setDef(Xcons.List(Xcode.FUNCTION_DEFINITION, accMain, mainIdList, mainDecls, mainBody));
  }
  
  private void createMain() {//throws ACCexception {
    Ident mainId = _env.findVarIdent("main");
    Xtype mainType = ((FunctionType)mainId.Type()).getBaseRefType();
    
    Ident argc = Ident.Local("argc", Xtype.intType);
    Ident argv = Ident.Local("argv", Xtype.Pointer(Xtype.Pointer(Xtype.charType)));
    XobjList args = Xcons.List(Xcode.ID_LIST, argc, argv);
    Ident accInitAll = _env.declExternIdent("acc_init_all", Xtype.Function(Xtype.voidType));
    Ident accMainId = _env.findVarIdent("_ACC_main");
    Ident accFinalizeAll = _env.declExternIdent("acc_finalize_all", Xtype.Function(Xtype.voidType));

    
    BlockList newFuncBody = Bcons.emptyBody();

    //newFuncBody.add(accInitAll.Call(ACCutil.getRefs(args)));
    
    if(mainType.equals(Xtype.voidType)){
      //newFuncBody.add(accMainId.Call(ACCutil.getRefs(args)));
      newFuncBody.add(accFinalizeAll.Call(null));
    }else{
      Ident r = Ident.Local("r", mainType);
      newFuncBody.addIdent(r);
      //newFuncBody.add(Xcons.Set(r.Ref(), accMainId.Call(args)));
      newFuncBody.add(accFinalizeAll.Call(null));
      newFuncBody.add(Xcons.List(Xcode.RETURN_STATEMENT, r.Ref()));
    }
    Ident newMainId = _env.declStaticIdent("main", Xtype.Function(mainType));
    _env.add(XobjectDef.Func(newMainId, args, null, newFuncBody.toXobject()));
    this.finalize();
  }
  
  private String getSourceBaseName(){
    String fullPath = _env.getSourceFileName();
    int dot = fullPath.lastIndexOf('.');
    int sep = fullPath.lastIndexOf('/');
    return fullPath.substring(sep + 1, dot);   // Delete extension and dirname　( "/tmp/hoge.c -> hoge" ).
  }
}

class FuncInfo{
  String funcName;
  Map<Integer, FuncArg> funcArgs;
  Map<Integer, FuncParam> funcParams;
  FuncInfo(String funcName){
    this.funcName = funcName;
    this.funcArgs = new HashMap<Integer, FuncArg>();
    this.funcParams = new HashMap<Integer, FuncParam>();
  }
  public void setParamInfo(int paramNum, Xcode type, boolean mustBeAllocated){ //for collee
    FuncParam funcParam = funcParams.get(paramNum);
    if(funcParam == null){
      funcParam = new FuncParam(paramNum, type, mustBeAllocated);
      funcParams.put(paramNum, funcParam);
    }else{
      if(funcParam.type != type){
        ACC.fatal("param type collision "+ funcParam.type + ":" + type);
      }else{
        funcParam.mustBeAllocated = funcParam.mustBeAllocated || mustBeAllocated;
      }
    }
  }
  public void setArgInfo(int argNum, Xcode type, boolean isAllocated){ //for coller
    FuncArg funcArg = funcArgs.get(argNum);
    if(funcArg == null){
      funcArg = new FuncArg(argNum, type, isAllocated);
      funcArgs.put(argNum, funcArg);
    }else{
      if(funcArg.type != type){
        ACC.fatal("arg type collision "+ funcArg.type + ":" + type);
      }else{
        funcArg.isAllocated = funcArg.isAllocated && isAllocated;
      }
    }
  }
}

class FuncParam{
  int paramNum;
  Xcode type;
  boolean mustBeAllocated;
  FuncParam(int paramNum, Xcode type, boolean mustBeAllocated){
    this.paramNum = paramNum;
    this.type = type;
    this.mustBeAllocated = mustBeAllocated;
  }
}

class FuncArg{
  int argNum;
  Xcode type;
  boolean isAllocated;
  FuncArg(int argNum, Xcode type, boolean isAllocated){
    this.argNum = argNum;
    this.type = type;
    this.isAllocated = isAllocated;
  }
  
  
  
}
