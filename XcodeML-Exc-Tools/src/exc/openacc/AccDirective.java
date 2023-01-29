/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.*;
import exc.object.*;

//
// superclass of data structure for each directive
//
public abstract class AccDirective {
  public static final String prop = "_ACC_DIRECTIVE";
  public final AccInformation _info;
  final XobjectDef _xobjDef;
  final PragmaBlock _pb;
  final ACCglobalDecl _decl;

  public AccDirective(ACCglobalDecl decl, AccInformation info){
    this(decl, info, null, null);
  }

  public AccDirective(ACCglobalDecl decl, AccInformation info, XobjectDef xobjDef){
    this(decl, info, xobjDef, null);
  }

  AccDirective(ACCglobalDecl decl, AccInformation info, PragmaBlock pb){
    this(decl, info, null, pb);
  }

  AccDirective(ACCglobalDecl decl, AccInformation info, XobjectDef xobjDef, PragmaBlock pb){
    _decl = decl;
    _info = info;
    _xobjDef = xobjDef;
    _pb = pb;
    ACC.debug(info.toString());
  }

  void analyze() throws ACCexception {
    if(ACC.debug_flag) System.out.println("AccDirective: analyze _info="+_info);
    _info.validate(this);
  }

  void setVarIdent(ACCvar var) throws ACCexception{
    if(ACC.debug_flag) System.out.println("setVarIdent var="+var);
    String symbol = var.getSymbol();
    Ident id = findVarIdent(symbol);
    if(id == null){
      throw new ACCexception("symbol '" + symbol + "' is not exist");
    }

    ACCvar parentVar = findParentVar(id);
    if(ACC.debug_flag) System.out.println("setVarIdent var="+var+" parentVar="+parentVar);
    if(parentVar != null && var != parentVar){
      var.setParent(parentVar);
    }else{
      var.setIdent(id);
      if(_info.getPragma() == ACCpragma.DECLARE) {
        setPropVar(id, var);
      }
    }
  }

  abstract void generate() throws ACCexception;
  abstract void rewrite() throws ACCexception;

  Ident findVarIdent(String symbol){
    Ident id;
    if(ACC.debug_flag)
      System.out.println("findVarIdent symbol="+symbol+" _pb_is_null="+(_pb==null)
                         +" _isloop="+_info.getPragma().isLoop());
    if(_pb == null) return _decl.findVarIdent(symbol);
    if(_info.getPragma().isLoop()){
      id = _pb.getBody().getHead().findVarIdent(symbol);
    } else 
      id = _pb.findVarIdent(symbol);
    if(ACC.debug_flag) System.out.println("findVarIdent ret id="+id);
    return id;
  }

  ACCvar findParentVar(Ident varId){
    {
      ACCvar var = getPropVar(varId);
      if(var != null) return var;
    }

    if(ACC.debug_flag) System.out.println("findPeranttVar search begin ... _pb.op="+_pb.Opcode());
    if(_pb != null) {
      for (Block b = _pb.getParentBlock(); b != null; b = b.getParentBlock()) {
        // System.out.println("findPeranttVar search bp="+b.Opcode());
        AccDirective directive = getPropDirective(b);
        if(directive == null) continue;
        //if(b.Opcode() != Xcode.ACC_PRAGMA) continue;
        AccInformation info = directive.getInfo();
        ACCvar var = info.findACCvar(varId.getSym());
        if(var != null && var.getId() == varId){
          return var;
        }
      }
      // System.out.println("findPeranttVar search end ...");
      
      ACCvar var = _decl.findACCvar(varId);
      if(var != null){
        return var;
      }
    }

    return null;
  }

  boolean isGlobal(){
    return _pb == null;
  }

  AccInformation getInfo(){
    return _info;
  }

  abstract boolean isAcceptableClause(ACCpragma clauseKind);

  boolean isIntExpr(Xobject expr) throws ACCexception{
    if(expr == null) return false;

    if(expr.Opcode() == Xcode.VAR){
      String varName = expr.getName();
      Ident varId = _pb.findVarIdent(varName);
      if(varId == null){
        throw new ACCexception("Symbol '" + varName + "' is not found");
      }
      if(varId.Type().isIntegral()){
        return true;
      }
    }else if(expr.isIntConstant()){
      return true;
    }

    //FIXME check complex expression
    return true;
  }

  boolean isSymbol(String sym) throws ACCexception{
    if(sym == null) return false;

    Ident id = findVarIdent(sym);
    return (id != null);
  }

//  private void fixXobject(Xobject x, Block b) throws ACCexception {
//    topdownXobjectIterator xIter = new topdownXobjectIterator(x);
//    for (xIter.init(); !xIter.end(); xIter.next()) {
//      Xobject xobj = xIter.getXobject();
//      if (xobj.Opcode() == Xcode.VAR) {
//        String name = xobj.getName();
//        Ident id = findVarIdent(b, name);
//        if (id == null) throw new ACCexception("'" + name + "' is not exist");
//        xIter.setXobject(id.Ref());
//      }
//    }
//  }

  private void setPropVar(Ident id, ACCvar var){
    id.setProp(ACCvar.prop, var);
  }
  private ACCvar getPropVar(Ident id){
    Object var = id.getProp(ACCvar.prop);
    if(var != null){
      return (ACCvar)var;
    }
    return null;
  }
  private AccDirective getPropDirective(Block b){
    Object dir = b.getProp(prop);
    if(dir != null){
      return (AccDirective)dir;
    }
    return null;
  }

  Xobject getAsyncExpr(){
    boolean isAsync = _info.hasClause(ACCpragma.ASYNC);

    if(! isAsync){
      return Xcons.IntConstant(ACC.ACC_ASYNC_SYNC);
    }

    Xobject expr = _info.getIntExpr(ACCpragma.ASYNC);

    if(expr == null){
      return Xcons.IntConstant(ACC.ACC_ASYNC_NOVAL);
    }

    return expr;
  }

  ACCpragma getDefaultVarAttr(){
    if(_pb != null) {
      for (Block b = _pb.getParentBlock(); b != null; b = b.getParentBlock()) {
        AccDirective directive = getPropDirective(b);
        if(directive == null) continue;
        AccInformation info = directive.getInfo();
        ACCpragma var_attr = info.getDefaultVarAttr();
        if(var_attr == ACCpragma.NONE) continue;
        return var_attr;
      }
    }
    return ACCpragma.PRESENT_OR_COPY; // default action
  }
}
