package exc.openacc;
import exc.object.*;

public class ACCvar {
  private Ident id;
  private ACCpragma atr;
  
  private ACCvar parent = null;
  
  //for data clause
  private boolean isPresent = false;
  private boolean isPresentOr = false;
  private boolean allocatesDeviceMemory = false;
  private boolean copyHostToDevice = false;
  private boolean copyDeviceToHost = false;
  private Ident deviceptr = null;
  private Ident hostDesc = null;
  
  //for reduction clause
  private boolean isReduction = false;
  
  //for parallel, kernels directive
  private boolean isFirstprivate = false;
  private boolean isPrivate = false;
  
  //for cache directive
  private boolean isCache = false;
  
  //for use_device clause
  private boolean isUse_device = false;
  
  
  ACCvar(Ident id, ACCpragma atr, ACCvar parent) throws ACCexception{
    this.id = id;
    this.atr = atr;
    this.parent = parent;
    
    if(parent != null){
      //inherit parent's attribute
      deviceptr = parent.deviceptr;
      isReduction = parent.isReduction;
      isFirstprivate = parent.isFirstprivate;
      isPrivate = parent.isPrivate;
      isCache = parent.isCache;
      isUse_device = parent.isUse_device;
    }
    
    switch(atr){
    case COPY:
      allocatesDeviceMemory = copyHostToDevice = copyDeviceToHost = true;
      break;
    case COPYIN:
      allocatesDeviceMemory = copyHostToDevice = true;
      break;
    case COPYOUT:
      allocatesDeviceMemory = copyDeviceToHost = true;
      break;
    case CREATE:
      allocatesDeviceMemory = true;
      break;
    case PRESENT:
      isPresent = true;
      break;
    case PRESENT_OR_COPY:
      isPresentOr = true;
      allocatesDeviceMemory = copyHostToDevice = copyDeviceToHost = true;
      break;
    case PRESENT_OR_COPYIN:
      isPresentOr = true;
      allocatesDeviceMemory = copyHostToDevice = true;
      break;
    case PRESENT_OR_COPYOUT:
      isPresentOr = true;
      allocatesDeviceMemory = copyDeviceToHost = true;
      break;
    case PRESENT_OR_CREATE:
      isPresentOr = true;
      allocatesDeviceMemory = true;
      break;
    case DEVICEPTR:
      deviceptr = id;
      break;
    case PRIVATE:
      isPrivate = true;
      break;
    case FIRSTPRIVATE:
      isFirstprivate = true;
      break;
    case CACHE:
      isCache = true;
      break;
    case HOST:
      copyDeviceToHost = true;
      break;
    case DEVICE:
      copyHostToDevice = true;
      break;
    default:
      if(atr.isReduction()){
        isReduction = true;
      }else{
        throw new ACCexception("var:"+id.getName()+", attribute:" + atr +" is not valid");
      }
        
    }
  }
  
  public String getName(){
    return id.getName();
  }
  
  public ACCpragma getAttribute(){
    return atr;
  }
  
  @Override
  public String toString(){
    return getName();
  }
  
  public boolean isPresent(){
    return isPresent;
  }

  public boolean isPresentOr(){
    return isPresentOr;
  }
  
  public boolean isReduction(){
    return isReduction;
  }
  public boolean isPrivate(){
    return isPrivate;
  }
  public boolean isFirstprivate(){
    return isFirstprivate;
  }
  public boolean isCache(){
    return isCache;
  }
  public boolean isUse_device(){
    return isUse_device;
  }
  
  public Ident getId(){
    return id;
  }
  
  public boolean allocatesDeviceMemory(){
    return allocatesDeviceMemory;
  }
  public boolean copiesHtoD(){
    return copyHostToDevice;
  }
  public boolean copiesDtoH(){
    return copyDeviceToHost;
  }
  public Ident getDevicePtr(){
    return deviceptr;
  }
  public void setDevicePtr(Ident devicePtr){
    this.deviceptr = devicePtr;
  }
  public void setHostDesc(Ident hostDesc){
    this.hostDesc = hostDesc;
  }
  public Ident getHostDesc(){
    return hostDesc;
  }
  public boolean isAllocated(){
    //return deviceptr != null;
    //return allocatesDeviceMemory();
    return (deviceptr != null ) || allocatesDeviceMemory;
  }
}
