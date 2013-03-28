package exc.openacc;
import exc.block.*;
import exc.object.*; 
import java.util.*;


public class ACCgpuManager {
  static final int maxBlockDim = 3;
  static final int maxThreadDim = 3;
  
  int availableBlockDim = maxBlockDim;
  int availableThreadDim = maxThreadDim;
  
  List<LoopExecInfo> loopExecInfos = new ArrayList<LoopExecInfo>();
  
  ACCgpuManager() {
  }
  
  public boolean addLoop(Iterator<ACCpragma> execModelIter, CforBlock... loops){
    ACCpragma execMethod = getExecMethod(execModelIter);
    switch(execMethod){
    case _BLOCK:
      return setAsBlock(loops);
    case _THREAD:
      return setAsThread(loops);
    case _BLOCK_THREAD:
      return setAsBlockThread(loops);
    case SEQ:
      return true;
    case _AUTO:
    {
      setAsAuto(loops);
      return true;
    }
    default:
      ACC.fatal("'" + execMethod.getName() + "' is not suppoerted");
    }
    return false;
  }
  
  private void setAsAuto(CforBlock... forBlocks){
    List<CforBlock> forBlockList = new ArrayList<CforBlock>();
    for(CforBlock forBlock : forBlocks) forBlockList.add(forBlock);
    loopExecInfos.add(new LoopExecInfo(forBlockList, ACCpragma._AUTO));
  }
  
  private boolean setAsBlock(CforBlock... forBlocks){
    if(availableBlockDim > 0){
      List<CforBlock> forBlockList = new ArrayList<CforBlock>();
      for(CforBlock forBlock : forBlocks) forBlockList.add(forBlock);
      loopExecInfos.add(new LoopExecInfo(forBlockList, ACCpragma._BLOCK));
      availableBlockDim--;
      return true;
    }
    return false;
  }
  
  private boolean setAsThread(CforBlock... forBlocks){
    if(availableThreadDim > 0){
      List<CforBlock> forBlockList = new ArrayList<CforBlock>();
      for(CforBlock forBlock : forBlocks) forBlockList.add(forBlock);
      loopExecInfos.add(new LoopExecInfo(forBlockList, ACCpragma._THREAD));
      availableThreadDim--;
      return true;
    }
    return false;
  }
  
  private boolean setAsBlockThread(CforBlock... forBlocks){
    if(availableBlockDim > 0 && availableThreadDim > 0){
      List<CforBlock> forBlockList = new ArrayList<CforBlock>();
      for(CforBlock forBlock : forBlocks) forBlockList.add(forBlock);
      loopExecInfos.add(new LoopExecInfo(forBlockList, ACCpragma._BLOCK_THREAD));
      availableBlockDim = availableThreadDim = Math.min(availableBlockDim, availableThreadDim) - 1;
      return true;
    }
    return false;
  }
  
  private ACCpragma getExecMethod(Iterator<ACCpragma> execModelIter){
    //XXX worker is not considered
    
    boolean gang = false;
    boolean worker = false;
    boolean vector = false;

    //if execModel is not specified, set it AUTO.
    if(! execModelIter.hasNext()){ 
      return ACCpragma._AUTO;
    }
    
    while(execModelIter.hasNext()){
      switch(execModelIter.next()){
      case GANG: gang = true;break;
      case WORKER: worker = true;break;
      case VECTOR: vector = true;break;
      case SEQ: break;
      default:
        ACC.fatal("getExecMethod error");
      }
    }
    
    if(gang){
      if(vector){
        return ACCpragma._BLOCK_THREAD;
      }else{
        return ACCpragma._BLOCK;
      }
    }else{
      if(vector){
        return ACCpragma._THREAD;
      }else{
        return ACCpragma.SEQ;
      }
    }
  }
  
  public void specifyExecModel(){
    //for execModel == _AUTO
    int num_auto = 0;
    for(LoopExecInfo loopExecInfo : loopExecInfos){
      if(loopExecInfo.method == ACCpragma._AUTO){
        num_auto++;
      }
    }

    ACCpragma prevExecMethod = null;
    
//    Iterator<LoopExecInfo> loopExecInfoIter = loopExecInfos.iterator();
//    for(; loopExecInfoIter.hasNext(); ){
//      LoopExecInfo loopExecInfo = loopExecInfoIter.next();
    for(LoopExecInfo loopExecInfo : loopExecInfos){
      if(loopExecInfo.method == ACCpragma._AUTO){
        ACCpragma method;
        if(num_auto > 1){
          if(prevExecMethod == null){
            method = ACCpragma._BLOCK;
          }else{
            method = prevExecMethod;            
          }
        }else{
          if(prevExecMethod == null){
            method = ACCpragma._BLOCK_THREAD;
          }else{
            method = ACCpragma._THREAD;
          }
        }
        
        if(method == ACCpragma._BLOCK || method == ACCpragma._BLOCK_THREAD){
          if(availableBlockDim == 0){
            method = ACCpragma.SEQ;
          }else{
            availableBlockDim--;
          }
        }
        if(method == ACCpragma._THREAD || method == ACCpragma._BLOCK_THREAD){
          if(availableThreadDim == 0){
            method = ACCpragma.SEQ;
          }else{
            availableThreadDim--;
          }
        }

        loopExecInfo.method = method;
        num_auto--;
      }

      prevExecMethod = loopExecInfo.method;
    }
    
  }
  
  public void distAxis(){
    int block_axis, thread_axis;
    block_axis = 2 - availableBlockDim;
    thread_axis = 2 - availableThreadDim;
    //XXX
    //block_axis = thread_axis = 2 - Math.min(availableBlockDim, availableThreadDim);
    Axis[] axisValues = Axis.values();
    
    for(LoopExecInfo loopExecInfo : loopExecInfos){
      switch(loopExecInfo.method){
      case _BLOCK:
        loopExecInfo.axis = axisValues[block_axis];
        block_axis--;
        break;
      case _THREAD:
        loopExecInfo.axis = axisValues[thread_axis];
        thread_axis--;
        break;
      case _BLOCK_THREAD:
        //XXX
        //if(block_axis != thread_axis) ACC.fatal("block_axis != thread_axis");
        loopExecInfo.axis = axisValues[block_axis];
        block_axis--;
        thread_axis--;
        break;
      default:
        ACC.fatal("unknown exec method");
      }
    }
  }
  
  public String getMethodName(CforBlock forBlock){
    for(LoopExecInfo loopExecInfo : loopExecInfos){
      if(loopExecInfo.forBlocks.contains(forBlock)){
        Axis axis = loopExecInfo.axis;
        switch(loopExecInfo.method){
        case _BLOCK:
          return "block_" + axis.getName();
        case _THREAD:
          return "thread_" + axis.getName();
        case _BLOCK_THREAD:
          return "block_thread_" + axis.getName();
        }
      }
    }
    return "";
  }
  
  public void finalize(){
    specifyExecModel();
    distAxis();
  }
  
  public XobjList getBlockThreadSize(){
    int usedBlockDim = 3 - availableBlockDim;
    int usedThreadDim = 3 - availableThreadDim;
        
    Xobject[] blockSize = {null, null, null};
    Xobject[] threadSize = {null, null, null};
    
    
    for(LoopExecInfo loopExecInfo : loopExecInfos){
      switch(loopExecInfo.method){
      case _THREAD:
      case _BLOCK_THREAD:
        ACCinfo info = ACCutil.getACCinfo(loopExecInfo.forBlocks.get(0));
        int idx = loopExecInfo.axis.ordinal();
        threadSize[idx] = info.getVectorLengthExp();
        break;
      }
    }
    for(int i = 0; i < usedThreadDim/*BlockDim*/; i++){
      if(threadSize[i] == null){
        threadSize[i] = Xcons.IntConstant(getDefaultThreadNum(usedThreadDim));
      }
    }

    
    for(LoopExecInfo loopExecInfo : loopExecInfos){
      switch(loopExecInfo.method){
      case _BLOCK:
      {
        ACCinfo info = ACCutil.getACCinfo(loopExecInfo.forBlocks.get(0));
        int idx = loopExecInfo.axis.ordinal();
        Xobject num_gangs = info.getNumGangsExp();
        if(num_gangs != null){
          blockSize[idx] = num_gangs;
        }else{
          blockSize[idx] = loopExecInfo.getTotalIterNum();
        }
      } break;
      case _BLOCK_THREAD:
      {
        ACCinfo info = ACCutil.getACCinfo(loopExecInfo.forBlocks.get(0));
        int idx = loopExecInfo.axis.ordinal();
        Xobject num_gangs = info.getNumGangsExp();
        if(num_gangs != null){
          blockSize[idx] = num_gangs;
        }else{
          blockSize[idx] = Xcons.binaryOp(Xcode.PLUS_EXPR, 
                                          Xcons.binaryOp(Xcode.DIV_EXPR, 
                                                         Xcons.binaryOp(Xcode.MINUS_EXPR, 
                                                                        loopExecInfo.getTotalIterNum(), 
                                                                        Xcons.IntConstant(1)),
                                                         threadSize[idx]),
                                          Xcons.IntConstant(1));
          //blockSize[idx] = ACCutil.foldIntConstant(blockSize[idx]);
        }
      } break;
      }
    }
    
    for(int i = 0; i < 3; i++){
      if(blockSize[i] == null) blockSize[i] = Xcons.IntConstant(1);
      if(threadSize[i] == null) threadSize[i] = Xcons.IntConstant(1);
    }

    return Xcons.List(Xcons.List(blockSize[0], blockSize[1], blockSize[2]),
                      Xcons.List(threadSize[0], threadSize[1], threadSize[2]));
  }
  private int getDefaultThreadNum(int totalThreadDim){
    switch(totalThreadDim){
    case 1: return 256;
    case 2: return 16;
    case 3: return 8;
    default: return 1;
    }
  }  
}

class LoopExecInfo{
  List<CforBlock> forBlocks;
  ACCpragma method;
  Axis axis;
  LoopExecInfo(List<CforBlock> forBlocks, ACCpragma method){
    this.forBlocks = forBlocks;
    this.method = method;
  }
  public Xobject getTotalIterNum(){
    Iterator<CforBlock> iter = forBlocks.iterator();
    
    return recTotalIterNum(iter);
  }
  private Xobject recTotalIterNum(Iterator<CforBlock> iter){
    CforBlock first;
    first = iter.next();
    if(iter.hasNext()){
      return Xcons.binaryOp(Xcode.MUL_EXPR, getIterNum(first), recTotalIterNum(iter));
    }else{
      return getIterNum(first);
    }
  }
  private Xobject getIterNum(CforBlock forBlock){
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

enum Axis{
  X, Y, Z;
  public String getName(){
    return toString().toLowerCase();
  }
}

