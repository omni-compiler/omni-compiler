package exc.openacc;


import exc.block.*;
import exc.object.*;
import java.util.*;


class AccManager {
  private final AccInformation _info;
  private final PragmaBlock _pb;
  private Xobject numGangs = null;
  private Xobject vectorLength = null;
  private final Map<CforBlock, LoopExecInfo> execMethodMap = new HashMap<CforBlock, LoopExecInfo>();

  private final int DEFAULT_THREAD_NUM = ACC.defaultVectorLength;

  //private final boolean useWorker = false;

  AccManager(AccInformation info, PragmaBlock pb){
    _info = info;
    _pb = pb;
  }

  public void analyze(){
    topdownBlockIterator blockIterator = new topdownBlockIterator(_pb);

    for (blockIterator.init(); !blockIterator.end(); blockIterator.next()) {
      Block b = blockIterator.getBlock();
      if (b.Opcode() != Xcode.ACC_PRAGMA) continue;

      AccDirective directive = (AccDirective)b.getProp(AccDirective.prop);
      AccInformation info = directive.getInfo(); //(AccInformation) b.getProp(AccInformation.prop);
      if(! info.getPragma().isLoop()) continue;
      CforBlock forBlock = (CforBlock)(b.getBody().getHead());
      EnumSet<ACCpragma> forBlockExecMethods = EnumSet.noneOf(ACCpragma.class);
      Xobject collapseNumExpr = null;

      if (info != null) {
        Set<ACCpragma> execModels = getExecModels(info);
        forBlockExecMethods.addAll(execModels);
        collapseNumExpr = info.getIntExpr(ACCpragma.COLLAPSE);
      }

      List<CforBlock> forBlocks = new ArrayList<CforBlock>();
      CforBlock fb = forBlock;

      int collapseNum = (collapseNumExpr == null) ? 1 : collapseNumExpr.getInt();
      for (int i = collapseNum; i > 0; i--) {
        forBlocks.add(fb);
        if (i > 1) {
          fb = AccLoop.findOutermostTightlyNestedForBlock(fb.getBody().getHead());
        }
      }
      execMethodMap.put(forBlock, new LoopExecInfo(forBlocks, forBlockExecMethods));
    }
  }

  private EnumSet<ACCpragma> getExecModels(AccInformation info){
    EnumSet<ACCpragma> execModels = EnumSet.noneOf(ACCpragma.class);
    if(info.hasClause(ACCpragma.GANG)) execModels.add(ACCpragma.GANG);
    if(info.hasClause(ACCpragma.WORKER)) execModels.add(ACCpragma.WORKER);
    if(info.hasClause(ACCpragma.VECTOR)) execModels.add(ACCpragma.VECTOR);
    if(info.hasClause(ACCpragma.SEQ)) execModels.add(ACCpragma.SEQ);

    if(execModels.isEmpty()){
      execModels.add(ACCpragma.AUTO);
    }
    return execModels;
  }

  public String getMethodName(CforBlock forBlock){
    LoopExecInfo loopExecInfo = execMethodMap.get(forBlock);

    EnumSet<ACCpragma> execMethods = loopExecInfo.methods;

//    if(!useWorker){
//      if(execMethods.size() == 1 && execMethods.contains(ACCpragma.WORKER)) {
//        return "";
//      }
//    }
    if(execMethods.isEmpty()){
      return "";
    }

    StringBuilder sb = new StringBuilder();
    if(execMethods.contains(ACCpragma.GANG)){
      sb.append("block_");
    }
//    if(useWorker) {
      if (execMethods.contains(ACCpragma.WORKER)) {
        sb.append("warp_");
      }
//    }
    if(execMethods.contains(ACCpragma.VECTOR)){
      sb.append("thread_");
    }
    sb.append('x');

    return new String(sb);

  }

  public EnumSet<ACCpragma> getMethodType(CforBlock forBlock){
    LoopExecInfo loopExecInfo = execMethodMap.get(forBlock);
    return loopExecInfo.methods;
  }

  public XobjList getBlockThreadSize(){
    Xobject bsx = getNumGangsExpr();
    Xobject tsx = getVectorLengthExpr();

    if(bsx == null) bsx = numGangs;
    if(tsx == null) tsx = vectorLength;

    for(CforBlock key : execMethodMap.keySet()) {
      LoopExecInfo loopExecInfo = execMethodMap.get(key);
      EnumSet<ACCpragma> execMethods = loopExecInfo.methods;

      if (execMethods.contains(ACCpragma.VECTOR)) {
        if (tsx == null) {
          tsx = Xcons.IntConstant(DEFAULT_THREAD_NUM);
        }
      }
    }
    if(tsx == null){
      tsx = Xcons.IntConstant(1);
    }

    for(CforBlock key : execMethodMap.keySet()) {
      LoopExecInfo loopExecInfo = execMethodMap.get(key);
      EnumSet<ACCpragma> execMethods = loopExecInfo.methods;

      if(execMethods.contains(ACCpragma.GANG)) {
        if (bsx == null) {
          if(execMethods.contains(ACCpragma.VECTOR)){
            bsx = Xcons.binaryOp(Xcode.PLUS_EXPR,
                    Xcons.binaryOp(Xcode.DIV_EXPR,
                            Xcons.binaryOp(Xcode.MINUS_EXPR,
                                    loopExecInfo.getTotalIterNum(),
                                    Xcons.IntConstant(1)),
                            tsx),
                    Xcons.IntConstant(1));
          }else {
            bsx = loopExecInfo.getTotalIterNum();
          }
        }
      }
    }


    if(bsx == null){
      bsx = Xcons.IntConstant(1);
    }

    Xobject xone = Xcons.IntConstant(1);

    //return Xcons.List(Xcons.List(bsx, xone, xone), Xcons.List(tsx, xone, xone));
    return Xcons.List(bsx, xone, tsx);
  }

  public void setNumGangs(Xobject numGangs){
    this.numGangs = numGangs;
  }

  public void setVectorLength(Xobject vectorLength){
    this.vectorLength = vectorLength;
  }

  private Xobject getNumGangsExpr(){
    Xobject expr = _info.getIntExpr(ACCpragma.NUM_GANGS);
    if(expr != null) return expr;
    return _info.getIntExpr(ACCpragma.GANG);
  }
  private Xobject getVectorLengthExpr(){
    Xobject expr = _info.getIntExpr(ACCpragma.VECT_LEN);
    if(expr != null) return expr;
    return _info.getIntExpr(ACCpragma.VECTOR);
  }

  class LoopExecInfo {
    final List<CforBlock> forBlocks;
    final EnumSet<ACCpragma> methods;

    LoopExecInfo(List<CforBlock> forBlocks, EnumSet<ACCpragma> methods) {
      this.forBlocks = forBlocks;
      this.methods = methods;
    }

    public Xobject getTotalIterNum() {
      Iterator<CforBlock> iter = forBlocks.iterator();

      return recTotalIterNum(iter);
    }

    private Xobject recTotalIterNum(Iterator<CforBlock> iter) {

      CforBlock first;
      first = iter.next();
      if (iter.hasNext()) {
        return Xcons.binaryOp(Xcode.MUL_EXPR, getIterNum(first), recTotalIterNum(iter));
      } else {
        return getIterNum(first);
      }
    }

    private Xobject getIterNum(CforBlock forBlock) {
      Xobject upper = forBlock.getUpperBound();
      Xobject lower = forBlock.getLowerBound();
      Xobject step = forBlock.getStep();

      return Xcons.binaryOp(Xcode.PLUS_EXPR,
              Xcons.binaryOp(Xcode.DIV_EXPR,
                      Xcons.binaryOp(Xcode.MINUS_EXPR,
                              Xcons.binaryOp(Xcode.MINUS_EXPR,
                                      upper,
                                      lower),
                              Xcons.IntConstant(1)),
                      step),
              Xcons.IntConstant(1));
    }
  }
}



