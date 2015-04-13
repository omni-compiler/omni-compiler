package exc.openacc;
import exc.object.*;
import java.util.*;



public class ACCvar {
  private Ident id;
  
  private boolean isSpecifiedDataAttribute = false;
  private EnumSet<Attribute> atrEnumSet = EnumSet.noneOf(Attribute.class);
  
  //for data clause
  private Ident deviceptr = null;
  private Ident hostDesc = null;
  
  //for reduction clause
  private ACCpragma reductionOp = null;
  
  //array dimension
  private int dim; // 0 means scalar
  private Xtype elementType;
  
  //for subarray
  private XobjList rangeList = Xcons.List();
  private boolean isSubarray = false;
  
  public static enum Attribute{
    isPresent,
    isPresentOr,
    create,
    delete,
    copyHostToDevice,
    copyDeviceToHost,
    isFirstprivate,
    isPrivate,
    isCache,
    isUseDevice,
    isReduction
  }
  
  ACCvar(Ident id, ACCpragma atr, ACCvar parent) throws ACCexception{
    this(id, null, atr, parent);
  }
  ACCvar(Ident id, XobjList subscripts, ACCpragma atr, ACCvar parent) throws ACCexception{
    this.id = id;
    
    if(atr != ACCpragma.USE_DEVICE){
      if(subscripts != null && !subscripts.isEmpty()){
        rangeList = makeRange(subscripts);
        isSubarray = true;
      }else{
        rangeList = makeRange(id.Type());
      }
    }
    
    dim = rangeList.Nargs();

    //System.out.println("var : " + getName());
    //System.out.println("rang=" + rangeList);
    //System.out.println("type=" + elementType);
    setAttribute(atr);

    if(parent != null && id == parent.getId()){
      //inherit parent's attribute
//      deviceptr = parent.deviceptr;
//      reductionOp = parent.reductionOp;
//      isFirstprivate = parent.isFirstprivate;
//      isPrivate = parent.isPrivate;
//      isCache = parent.isCache;
//      isUse_device = parent.isUse_device;
    }
  }
  
  public void setAttribute(ACCpragma atr) throws ACCexception{
    if(atr.isDataClause() && isSpecifiedDataAttribute){
      ACC.fatal("ACCvar: " + id.getName() + " is already specified data attribute");
    }
    
    switch(atr){
    case COPY:
      atrEnumSet.add(Attribute.create);
      atrEnumSet.add(Attribute.copyHostToDevice);
      atrEnumSet.add(Attribute.copyDeviceToHost);
      break;
    case COPYIN:
      atrEnumSet.add(Attribute.create);
      atrEnumSet.add(Attribute.copyHostToDevice);
      break;
    case COPYOUT:
      atrEnumSet.add(Attribute.create);
      atrEnumSet.add(Attribute.copyDeviceToHost);
      break;
    case CREATE:
      atrEnumSet.add(Attribute.create);
      break;
    case DELETE:
      atrEnumSet.add(Attribute.delete);
      break;
    case PRESENT:
      atrEnumSet.add(Attribute.isPresent);
      break;
    case PRESENT_OR_COPY:
      atrEnumSet.add(Attribute.isPresentOr);
      atrEnumSet.add(Attribute.create);
      atrEnumSet.add(Attribute.copyHostToDevice);
      atrEnumSet.add(Attribute.copyDeviceToHost);
      break;
    case PRESENT_OR_COPYIN:
      atrEnumSet.add(Attribute.isPresentOr);
      atrEnumSet.add(Attribute.create);
      atrEnumSet.add(Attribute.copyHostToDevice);
      break;
    case PRESENT_OR_COPYOUT:
      atrEnumSet.add(Attribute.isPresentOr);
      atrEnumSet.add(Attribute.create);
      atrEnumSet.add(Attribute.copyDeviceToHost);
      break;
    case PRESENT_OR_CREATE:
      atrEnumSet.add(Attribute.isPresentOr);
      atrEnumSet.add(Attribute.create);
      break;
    case DEVICEPTR:
      deviceptr = id;
      break;
    case PRIVATE:
      atrEnumSet.add(Attribute.isPrivate);
      break;
    case FIRSTPRIVATE:
      atrEnumSet.add(Attribute.isFirstprivate);
      break;
    case CACHE:
      atrEnumSet.add(Attribute.isCache);
      break;
    case HOST:
      atrEnumSet.add(Attribute.copyDeviceToHost);
      break;
    case DEVICE:
      atrEnumSet.add(Attribute.copyHostToDevice);
      break;
    case USE_DEVICE:
      atrEnumSet.add(Attribute.isUseDevice);
      break;
    default:
      if(atr.isReduction()){
        atrEnumSet.add(Attribute.isReduction);
        reductionOp = atr;
      }else{
        throw new ACCexception("var:"+id.getName()+", attribute:" + atr +" is not valid");
      } 
    }
  }
  
  public String getName(){
    return id.getName();
  }

  @Override
  public String toString(){
    return getName();
  }
  
  
  public boolean isPresent(){
    return atrEnumSet.contains(Attribute.isPresent);
  }
  public boolean isPresentOr(){
    return atrEnumSet.contains(Attribute.isPresentOr);
  }
  public boolean allocatesDeviceMemory(){
    return atrEnumSet.contains(Attribute.create);
  }
  public boolean copiesHtoD(){
    return atrEnumSet.contains(Attribute.copyHostToDevice);
  }
  public boolean copiesDtoH(){
    return atrEnumSet.contains(Attribute.copyDeviceToHost);
  }
  public boolean isPrivate(){
    return atrEnumSet.contains(Attribute.isPrivate);
  }
  public boolean isFirstprivate(){
    return atrEnumSet.contains(Attribute.isFirstprivate);
  } 
  public boolean isReduction(){
    return atrEnumSet.contains(Attribute.isReduction);
  }
  public boolean isCache(){
    return atrEnumSet.contains(Attribute.isCache);
  }

  public boolean is(Attribute attr)
  {
    return atrEnumSet.contains(attr);
  }
  
  
  public Ident getId(){
    return id;
  }
  public boolean isUse_device(){
    return atrEnumSet.contains(Attribute.isUseDevice);
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
    //return (deviceptr != null ) || allocatesDeviceMemory;
    return (deviceptr != null ) || atrEnumSet.contains(Attribute.create);
  }
  public ACCpragma getReductionOperator(){
    return reductionOp;
  }
  public boolean contains(XobjList subscripts){
    return true;
    //FIXME implement!
  }
  
  private void addRange(XobjList rangeList, Xobject range, ArrayType arrayType) throws ACCexception{
    Xobject lower, length;
    if(range.Opcode() != Xcode.LIST){ //scalar
      lower = range;
      length = Xcons.IntConstant(1);
    }else{ //sub
      lower = range.getArg(0);
      length = range.getArgOrNull(1);
      if(length == null){
        if(arrayType != null){
          length = Xcons.binaryOp(Xcode.MINUS_EXPR, getArraySize(arrayType), lower);  
        }else{
          throw new ACCexception("length is unspecified");
        }
      }
    }
    if(arrayType != null && isCorrectRange(arrayType, lower, length) == false){
      throw new ACCexception("array bound exceeded : " + getName());
    }
    rangeList.add(Xcons.List(lower, length));
  }
  
  private XobjList makeRange(XobjList subscript) throws ACCexception{
    XobjList rangeList = Xcons.List();
    Xtype type = id.Type();
    XobjArgs args = subscript.getArgs();
    while(args != null){
      Xobject range = args.getArg();
      switch(type.getKind()){
      case Xtype.ARRAY:
        ArrayType arrayType = (ArrayType)type;
        addRange(rangeList, range, arrayType);
        type = arrayType.getRef();
        break;
      case Xtype.POINTER:
        if(range.Opcode() == Xcode.LIST){
          addRange(rangeList, range, null);
          type = type.getRef();
          if(false){
            if(type.isBasic() || type.isEnum() || type.isStruct()){
              for(; args != null; args = args.nextArgs()){
                range = args.getArg();
                addRange(rangeList, range, null);
              }
            }
          }
          break;
        }
        throw new ACCexception("unshaped pointer");
      default:
        throw new ACCexception("too many subscripts");   
      }
      if(args != null) args = args.nextArgs();
    }
    if(type.isArray()){
      throw new ACCexception("too few subscripts");
    }
    this.elementType = type;
    return rangeList;
  }
  
  private boolean isCorrectRange(ArrayType arrayType, Xobject lower, Xobject length){
    Xobject size = getArraySize(arrayType);
    size = ACCutil.foldIntConstant_mod(size);
    if(! size.isIntConstant()) return true;
    int sizeInt = size.getInt();
    
    Xobject lower2 = ACCutil.foldIntConstant_mod(lower);
    Xobject length2 = ACCutil.foldIntConstant_mod(length);
    int lowerInt = 0;
    int lengthInt = 1;
    if(lower2.isIntConstant()){
      lowerInt = lower2.getInt();
    }
    if(length2.isIntConstant()){
      lengthInt = length2.getInt();
    }
    
    if(lowerInt < 0 || lengthInt < 1) return false;
    if(lowerInt + lengthInt > sizeInt) return false;
    
    return true;
  }
  
  private XobjList makeRange(Xtype type){
    XobjList rangeList = Xcons.List();
    
    while(true){
      switch(type.getKind()){
      case Xtype.ARRAY:
      {
        ArrayType arrayType = (ArrayType)type;
        rangeList.add(Xcons.List(Xcons.IntConstant(0), getArraySize(arrayType)));
        type = arrayType.getRef();
      } break;
      case Xtype.BASIC:
      case Xtype.STRUCT:
      case Xtype.ENUM:
        this.elementType = type;
        return rangeList;
      case Xtype.POINTER:
        ACC.warning("pointer reference was treated as array reference in '" + getName() +"'");
        rangeList.add(Xcons.List(Xcons.IntConstant(0), Xcons.IntConstant(1)));
        type = type.getRef();
        break;
      default:
        ACC.fatal("unsupposed type");
      }
    }
  }
  
  private Xobject getArraySize(ArrayType arrayType){
    long arraySize = arrayType.getArraySize();
    if (arraySize <= 0){
      return arrayType.getArraySizeExpr();
    }
    if(arraySize > Integer.MAX_VALUE){
      return Xcons.LongLongConstant(0,arraySize);
    }
    else{
      return Xcons.IntConstant((int)arraySize);
    }
  }
  
  public Xobject getAddress() throws ACCexception{
    Xobject addrObj = null;
    Xtype varType = id.Type();

    switch (varType.getKind()) {
    case Xtype.BASIC:
    case Xtype.STRUCT:
    case Xtype.UNION:
      addrObj = id.getAddr();
      break;
    case Xtype.POINTER:
    {
      if(isSubarray()){
        addrObj = id.Ref();
      }else{
        addrObj = id.getAddr();
      }
      break;
    }
    case Xtype.ARRAY:
    {
      ArrayType arrayVarType = (ArrayType)varType;
      switch (arrayVarType.getArrayElementType().getKind()) {
      case Xtype.BASIC:
      case Xtype.STRUCT:
      case Xtype.UNION:
        break;
      default:
        throw new ACCexception("array '" + getName() + "' has a wrong data type for acc data");
      }

      addrObj = id.Ref();
      break;
    }
    default:
      throw new ACCexception("'" + getName() + "' has a wrong data type for acc data");
    }
    return addrObj;
  }

  public Xobject getSize() throws ACCexception{
    Xobject size = Xcons.SizeOf(elementType);
    for(Xobject x : rangeList){
      size = Xcons.binaryOp(Xcode.MUL_EXPR, size, (XobjList)x.left());
    }
    return size;
  }
  
  /////////////////
  
  public boolean conllidesWith(XobjList subscripts){
    int dim = Math.max(this.rangeList.Nargs(), subscripts.Nargs());
    for(int i=0;i<dim;i++){
      Xobject s1 = this.rangeList.getArg(i);
      Xobject s2 = subscripts.getArg(i);
      long low1=0,len1=1,low2=0,len2=1;
      try{
        if(s1.Opcode() != Xcode.LIST){ //s1 is single
          low1 = toLong(s1);
          //len1 = 1;
        }else{
          low1 = toLong(s1.getArg(0));
          len1 = toLong(s1.getArgOrNull(1));
        }
        if(s2.Opcode() != Xcode.LIST){ //s2 is single
          low2 = toLong(s2);
          //len1 = 1;
        }else{
          low2 = toLong(s2.getArg(0));
          len2 = toLong(s2.getArgOrNull(1));
        }
      }catch(ACCexception e){
        return false;
      }
      if(hasIntersect(low1, len1, low2, len2)) return true;
    }
    return false;
  }
  private long toLong(Xobject x) throws ACCexception{
    if(x == null){
      throw new ACCexception("null");
    }
    if(x.isIntConstant()){
      return (long)x.getInt();
    }else if(x.Opcode()== Xcode.LONGLONG_CONSTANT){
      return (long)x.getLong();
    }throw new ACCexception("not constant");

  }
  private boolean hasIntersect(long low1, long len1, long low2, long len2){
    if(low1 < low2){
      return (low1+len1 > low2);
    }else{
      return (low2+len2 > low1);
    }
  }
  
  public XobjList getSubscripts(){
    return rangeList;
  }
  
  public boolean isSubarray(){
    return isSubarray;
  }
  
  public Xtype getElementType(){
    return this.elementType;
  }

  public int getDim(){
    return this.dim;
  }
}


