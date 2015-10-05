package exc.openacc;

import exc.block.*;
import exc.object.*;

class AccHostData extends AccDirective {
  AccHostData(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }

  /*
  @Override
  void analyze() throws ACCexception {
    setVarIdents();
  }
  */

  @Override
  void generate() throws ACCexception {
  }

  @Override
  void rewrite() throws ACCexception {
    for(ACCvar var : _info.getACCvarList()){
      rewriteVar(var);
    }

    _pb.replace(Bcons.COMPOUND(_pb.getBody()));
  }

  //FIXME
  private void rewriteVar(ACCvar var){
    String hostName = var.getName();
    Xobject deviceAddr = var.getDevicePtr().Ref();

    BasicBlockExprIterator iter = new BasicBlockExprIterator(_pb.getBody());
    for (iter.init(); !iter.end(); iter.next()) {
      //XobjectIterator exprIter = new topdownXobjectIterator(iter.getExpr());
      XobjectIterator exprIter = new bottomupXobjectIterator(iter.getExpr());
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
	if(x == null) continue;
        switch (x.Opcode()) {
        case VAR:
        {
          String symbol = x.getSym();
          if(! symbol.equals(hostName)) break;

          Xtype varType = var.getId().Type();
          Xobject new_x;
          if(varType.isPointer()){
            new_x = Xcons.Cast(Xtype.Pointer(x.Type().getRef()), deviceAddr);
          }else{
            new_x = Xcons.PointerRef(Xcons.Cast(Xtype.Pointer(varType), deviceAddr));
          }
          exprIter.setXobject(new_x);
        }break;
        case ARRAY_ADDR:
        {
          String arrayName = x.getName();
          if(! arrayName.equals(hostName)) break;
          Xobject newObj = Xcons.Cast(Xtype.Pointer(x.Type().getRef()), deviceAddr);
          exprIter.setXobject(newObj);
        }break;
        case ARRAY_REF:
        {
          Xobject arrayAddr = x.getArg(0);
          if(arrayAddr.Opcode() == Xcode.ARRAY_ADDR)break;
          exprIter.setXobject(convertArrayRef(x));
        } break;
        default:
        }
      }
    }
  }

  private Xobject convertArrayRef(Xobject x)
  {
    if(x.Opcode() != Xcode.ARRAY_REF) return x;
    Xobject arrayAddr = x.getArg(0);
    XobjList indexList = (XobjList)(x.getArg(1));

    Xobject result = arrayAddr;
    for(Xobject idx : indexList){
      result = Xcons.PointerRef(Xcons.binaryOp(Xcode.PLUS_EXPR, result, idx));
    }
    return result;
  }

  boolean isAcceptableClause(ACCpragma clauseKind) {
    return clauseKind == ACCpragma.USE_DEVICE;
  }
}
