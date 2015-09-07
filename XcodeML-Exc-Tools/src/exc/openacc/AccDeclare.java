package exc.openacc;

import exc.block.*;
import exc.object.*;

class AccDeclare extends AccData{
  AccDeclare(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }
  AccDeclare(ACCglobalDecl decl, AccInformation info) {
    super(decl, info);
  }

  @Override
  void analyze() throws ACCexception{
    super.analyze();
    if(isGlobal()){
      for(ACCvar var : _info.getDeclarativeACCvarList()){
        _decl.addACCvar(var);
      }
    }
  }
  
  @Override
  void rewrite() throws ACCexception{
    BlockList initBody = Bcons.emptyBody();
    BlockList finalizeBody = Bcons.emptyBody();

    for(Block b : initBlockList) initBody.add(b);
    for(Block b : copyinBlockList) initBody.add(b);
    for(Block b : copyoutBlockList) finalizeBody.add(b);
    for(Block b : finalizeBlockList) finalizeBody.add(b);

    Block beginBlock = Bcons.COMPOUND(initBody);
    Block endBlock = Bcons.COMPOUND(finalizeBody);

    if(isGlobal()) {
      XobjList id_list = (XobjList)_decl.getEnv().getGlobalIdentList();
      id_list.mergeList(idList);
      //_decl.getEnv().setIdentList(id_list);
      _decl.addGlobalConstructor(beginBlock.toXobject());
      _decl.addGlobalDestructor(endBlock.toXobject());
    }else{
      BlockList resultBody = Bcons.emptyBody(idList, null);
      resultBody.add(beginBlock);
      resultBody.add(Bcons.COMPOUND(_pb.getBody()));
      resultBody.add(endBlock);
      _pb.replace(Bcons.COMPOUND(resultBody));
    }
  }

  boolean isAcceptableClause(ACCpragma clauseKind) {
    switch (clauseKind){
    case IF:
    case ASYNC:
      return true;
    default:
      return clauseKind.isDataClause();
    }
  }
}
