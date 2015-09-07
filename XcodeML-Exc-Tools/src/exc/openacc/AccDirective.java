package exc.openacc;

import exc.block.*;
import exc.object.*;

abstract class AccDirective {
  public static final String prop = "_ACC_DIRECTIVE";
  final AccInformation _info;
  final PragmaBlock _pb;
  final ACCglobalDecl _decl;
  AccDirective(ACCglobalDecl decl, AccInformation info){
    this(decl, info, null);
  }
  AccDirective(ACCglobalDecl decl, AccInformation info, PragmaBlock pb){
    _decl = decl;
    _info = info;
    _pb = pb;
    ACC.debug(info.toString());
  }

  void analyze() throws ACCexception {
    _info.validate(this);
  }

  void setVarIdent(ACCvar var) throws ACCexception{
    String symbol = var.getSymbol();
    Ident id = findVarIdent(symbol);
    if(id == null){
      throw new ACCexception("symbol '" + symbol + "' is not exist");
    }
    //var.setIdent(id);
    ACCvar parentVar = findParentVar(id);
    if(parentVar != null){
      var.setParent(parentVar);
    }else{
      var.setIdent(id);
    }
  }

  abstract void generate() throws ACCexception;
  abstract void rewrite() throws ACCexception;

  Ident findVarIdent(String symbol){
    if(_pb == null) return _decl.findVarIdent(symbol);
    if(_info.getPragma().isLoop()){
      return _pb.getBody().getHead().findVarIdent(symbol);
    }
    return _pb.findVarIdent(symbol);
  }

  ACCvar findParentVar(Ident varId){
    if(_pb != null) {
      for (Block b = _pb.getParentBlock(); b != null; b = b.getParentBlock()) {
        if(b.Opcode() != Xcode.ACC_PRAGMA) continue;
        AccInformation info = ((AccDirective)b.getProp(prop)).getInfo();
        ACCvar var = info.findACCvar(varId.getSym());
        if(var != null && var.getId() == varId){
          return var;
        }
      }
      
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
}
