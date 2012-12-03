package exc.openacc;
import exc.object.Xobject;


public enum ACCpragma {
  //directive
  PARALLEL,
  KERNELS,
  DATA,
  HOST_DATA,
  LOOP,
  CACHE,
  PARALLEL_LOOP,
  KERNELS_LOOP,
  DECLARE,
  UPDATE,
  WAIT,

  
  //general
  IF,
  ASYNC,
  
  //accelerator clause
  NUM_GANGS,
  NUM_WORKERS,
  VECT_LEN,
  PRIVATE,
  FIRSTPRIVATE,
  REDUCTION_PLUS,
  REDUCTION_MUL,
  REDUCTION_MIN,
  REDUCTION_MAX,
  REDUCTION_BITAND,
  REDUCTION_BITOR,
  REDUCTION_BITXOR,
  REDUCTION_LOGAND,
  REDUCTION_LOGOR,
  
  //host_data clause
  USE_DEVICE,
  
  //data clause
  DEVICEPTR,
  COPY,
  COPYIN,
  COPYOUT,
  CREATE,
  PRESENT,
  PRESENT_OR_COPY,
  PRESENT_OR_COPYIN,
  PRESENT_OR_COPYOUT,
  PRESENT_OR_CREATE,
  
  //loop clause
  COLLAPSE,
  GANG,
  WORKER,
  SEQ,
  VECTOR,
  INDEPENDENT,
  
  //declare clause
  DEVICE_RESIDENT,
  
  //update clause
  HOST,
  DEVICE,
  
  //internal
  _BLOCK,
  _BLOCK_THREAD,
  _WARP,
  _VECTORTHREAD,
  _THREAD,
  ;
  
  private String name = null;
  
  public String getName() {
    if (name == null) name = toString().toLowerCase();
    return name;
  }

  public static ACCpragma valueOf(Xobject x) {
    return valueOf(x.getString());
  }
  
  public static boolean isDataClause(ACCpragma clause){
    switch(clause){
    case COPY:  
    case COPYIN:  
    case COPYOUT:
    case CREATE:      
    case PRESENT:
    case PRESENT_OR_COPY:
    case PRESENT_OR_COPYIN:
    case PRESENT_OR_COPYOUT:
    case PRESENT_OR_CREATE:
    case DEVICEPTR:
      return true;
    default:
      return false;
    }
  }
  
  public boolean isDataClause(){
    return isDataClause(this);
  }
  
  public boolean isDirective(){
    switch(this){
    case PARALLEL:
    case KERNELS:
    case DATA:
    case HOST_DATA:
    case CACHE:
    case DECLARE:
    case WAIT:
      return true;
    default:
      return false;
    }
  }
  
  public boolean isReduction(){
    switch(this){
    case REDUCTION_PLUS:
    case REDUCTION_MUL:
    case REDUCTION_MIN:
    case REDUCTION_MAX:
    case REDUCTION_BITAND:
    case REDUCTION_BITOR:
    case REDUCTION_BITXOR:
    case REDUCTION_LOGAND:
    case REDUCTION_LOGOR:
      return true;
    default:
      return false;
    }
  }
}
