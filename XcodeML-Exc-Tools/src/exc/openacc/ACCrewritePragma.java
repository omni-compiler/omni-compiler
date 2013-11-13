package exc.openacc;
import xcodeml.util.XmOption;
import exc.block.*;
import exc.object.*;

import java.util.*;

public class ACCrewritePragma implements XobjectDefVisitor{
  private ACCglobalDecl _globalDecl;
  
  public ACCrewritePragma(ACCglobalDecl globalDecl){
    // FIXME current implementation only supports C language
    if (!XmOption.isLanguageC())
      ACC.fatal("current version only supports C language.");

    _globalDecl = globalDecl;
  }
  
  public void finalize() {
    _globalDecl.finalize();
  }

  public void doDef(XobjectDef def) {
    rewrite(def);
  }
  
  private void rewrite(XobjectDef def) {
    if (def.isFuncDef()) {
      FuncDefBlock fd = new FuncDefBlock(def);
      rewriteLocalPragma_new(fd);
    }else{
      rewriteGlobalPragma(def);
      return;
    }
  }
  
  private void rewriteLocalPragma(FuncDefBlock def){
    FunctionBlock fb = def.getBlock();
    XobjectDef currentDef = def.getDef();

    BlockIterator i = new topdownBlockIterator(fb);
    for (i.init(); !i.end(); i.next()) {
      Block b = i.getBlock();
      if (b.Opcode() ==  Xcode.ACC_PRAGMA) {
        PragmaBlock pb = (PragmaBlock)b;
//        try {
          ACCinfo info = ACCutil.getACCinfo(pb);
          if(info != null){
//            List<Ident> idList = info.getIdList();
            XobjList idList = info.getIdList();
            XobjList declList = info.getDeclList();
//            List<Block> beginBlockList = info.getBeginBlockList();
//            List<Block> endBlockList = info.getEndBlockList();
            Block beginBlock = info.getBeginBlock();
            Block endBlock = info.getEndBlock();
            Block replaceBlock = info.getReplaceBlock();

            BlockList newBody = Bcons.emptyBody();
//            if(idList != null){
//              for(Ident id : idList) newBody.addIdent(id);
//            }
            newBody.setIdentList(idList);
//            if(beginBlockList != null){
//              for(Block bb : beginBlockList) newBody.add(bb);
//            }
            newBody.setDecls(declList);
            if(beginBlock != null){
              newBody.add(beginBlock);
            }
            if(replaceBlock != null){
              newBody.add(replaceBlock);
            }else{
              newBody.add(Bcons.COMPOUND(pb.getBody()));
            }
//            if(endBlockList != null){
//              for(Block eb : endBlockList) newBody.add(eb);
//            }
            if(endBlock != null){
              newBody.add(endBlock);
            }

            Block newBlock = Bcons.COMPOUND(newBody);
            pb.replace(newBlock);
          }
//        } catch (ACCexception e) {
//          ACC.error(pb.getLineNo(), e.getMessage());
//        }
      }else if(b.Opcode() == Xcode.PRAGMA_LINE){
        PragmaBlock pb = (PragmaBlock)b;
        ACC.error(pb.getLineNo(), "unknown pragma : " + pb.getClauses());
      }
    }
    def.Finalize();
  }
  
  private void rewriteGlobalPragma(XobjectDef def){
    //XXX need to implement

    Xobject x = def.getDef();
    if(x.Opcode() == Xcode.ACC_PRAGMA){
      // for global declare
      String pragmaName = x.getArg(0).getString();
      switch(ACCpragma.valueOf(pragmaName)){  
      case DECLARE:
        ACCtranslateDeclare declare = new ACCtranslateDeclare(x);
        declare.rewrite();
        break;
      default:
          
      }
    }else if(x.Opcode() == Xcode.PRAGMA_LINE){
      ACC.error(x.getLineNo(), "unknown pragma : " + x);
    }
  }
  
  
  private void rewriteLocalPragma_new(FuncDefBlock def){
    FunctionBlock fb = def.getBlock();
    //XobjectDef currentDef = def.getDef();

    BlockIterator i = new topdownBlockIterator(fb);
    for (i.init(); !i.end(); i.next()) {
      Block b = i.getBlock();
      if (b.Opcode() == Xcode.ACC_PRAGMA) {
          PragmaBlock pb = (PragmaBlock)b;
          ACCinfo info = ACCutil.getACCinfo(pb);
          
          if(info != null){
            if(info.getPragma() == ACCpragma.HOST_DATA){
              ACCtranslateHostData hostdata = new ACCtranslateHostData(pb);
              hostdata.rewrite();
              Block newBlock = Bcons.COMPOUND(pb.getBody());
              pb.replace(newBlock);
            }else if(info.getPragma() == ACCpragma.DECLARE){
              ACCtranslateDeclare declare = new ACCtranslateDeclare(pb);
              declare.rewrite();
            }else{
              XobjList idList = info.getIdList();
              XobjList declList = info.getDeclList();
              Block beginBlock = info.getBeginBlock();
              Block endBlock = info.getEndBlock();
              Block replaceBlock = info.getReplaceBlock();

              BlockList newBody = Bcons.emptyBody();
              newBody.setIdentList(idList);
              newBody.setDecls(declList);
              if(beginBlock != null){
                newBody.add(beginBlock);
              }
              if(replaceBlock != null){
                newBody.add(replaceBlock);
              }else{
                newBody.add(Bcons.COMPOUND(pb.getBody()));
              }
              if(endBlock != null){
                newBody.add(endBlock);
              }

              Block newBlock = Bcons.COMPOUND(newBody);
              pb.replace(newBlock);
            }
          }
          def.Finalize();
          rewriteLocalPragma_new(def);
          return;
      }else if(b.Opcode() == Xcode.PRAGMA_LINE){
        PragmaBlock pb = (PragmaBlock)b;
        ACC.error(pb.getLineNo(), "unknown pragma : " + pb.getClauses());
      }
    }
    //def.Finalize();
  }
}
