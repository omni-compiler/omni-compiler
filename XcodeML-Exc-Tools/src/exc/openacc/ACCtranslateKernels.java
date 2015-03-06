package exc.openacc;

import java.util.*;

import exc.block.*;
import exc.object.*;


public class ACCtranslateKernels {

  
  private PragmaBlock pb;
  private ACCinfo kernelsInfo;
  private ACCglobalDecl globalDecl;

  //private List<ACCvar> reductionVars;
 
  public ACCtranslateKernels(PragmaBlock pb) {
    this.pb = pb;
    this.kernelsInfo = ACCutil.getACCinfo(pb);
    if(this.kernelsInfo == null){
      ACC.fatal("can't get info");
    }
    this.globalDecl = this.kernelsInfo.getGlobalDecl();
  }
  
  public void translate() throws ACCexception{
    ACC.debug("translate kernels");
    
    if(kernelsInfo.isDisabled()){
      return;
    }
    
    //divide to kernels -> list of kernel head block 
    List<List<Block>> kernelBodyList = divideBlocksBetweenKernels(pb);
    
    //analyze and complete clause for kernel
    //determine arguments for launchFunc
    List<ACCgpuKernel> kernelList = new ArrayList<ACCgpuKernel>();
    for(List<Block> kernelBody : kernelBodyList){
      ACCgpuKernel gpuKernel = new ACCgpuKernel(kernelsInfo, kernelBody);
      kernelList.add(gpuKernel);
      gpuKernel.analyze();
    }
    
    
    //get intersection of readonly id set in each kernel
    Set<Ident> readOnlyOuterIdSet = collectReadOnlyOuterIdSet(kernelList);
    for(ACCgpuKernel gpuKernel: kernelList){
      gpuKernel.setReadOnlyOuterIdSet(readOnlyOuterIdSet);
    }
    
    
    //set unspecified var's attribute from outerIdSet
    Set<Ident> outerIdSet = new HashSet<Ident>();
    for(ACCgpuKernel gpuKernel : kernelList){
      List<Ident> kernelOuterId = gpuKernel.getOuterIdList();
      outerIdSet.addAll(kernelOuterId);
    }
    for(Ident id : outerIdSet){
      //if(kernelsInfo.isVarFirstprivate(id.getName())) continue; //only parallel construct
      //if(kernelsInfo.isVarPrivate(id.getName())) continue; //only parallel construct
      if(readOnlyOuterIdSet.contains(id) && !id.Type().isArray()) continue; //scalar variable && readOnly
      if(kernelsInfo.isVarAllocated(id.getName())) continue; //if var is already allocated 
      ////if(kernelsInfo.getDevicePtr(id.getName()) != null) continue;
      if(kernelsInfo.isVarReduction(id.getName())) continue;
      
      ACC.warning("Line" + pb.getLineNo() + ":'" + id.getName() + "' is treated as pcopy");
      kernelsInfo.declACCvar(id.getName(), ACCpragma.PRESENT_OR_COPY);
    }
    
    //translate data
    ACCtranslateData dataTranslator = new ACCtranslateData(pb);
    dataTranslator.translate();
    
    //make kernels list of block(kernel call , sync) 
    BlockList replaceBody = Bcons.blockList();
    for(ACCgpuKernel gpuKernel : kernelList){
      Block kernelCallBlock = gpuKernel.makeLaunchFuncCallBlock();
      replaceBody.add(kernelCallBlock);
    }
    
    //set replace block
    kernelsInfo.setReplaceBlock(Bcons.COMPOUND(replaceBody));
  }
  
  private List<List<Block>> divideBlocksBetweenKernels(PragmaBlock pb) {
    List<List<Block>> blockListList = new ArrayList<List<Block>>();
    
    if(kernelsInfo.pragma == ACCpragma.KERNELS){
      BlockList pbBody = pb.getBody();
      for(Block b = pbBody.getHead(); b != null; b = b.getNext()){
        List<Block> blockList = new ArrayList<Block>();
        blockList.add(b);
        blockListList.add(blockList);
      }
    }else{ //ACCpragma.KERNELS_LOOP
      List<Block> blockList = new ArrayList<Block>();
      blockList.add(pb);
      blockListList.add(blockList);
    }
    
    return blockListList;
  }

  /*
  public void analyzeKernelBody(List<Block> kernelBody){
    if(kernelBody.size() == 1){
      
    }
  }*/
  
  private Set<Ident> collectReadOnlyOuterIdSet(List<ACCgpuKernel> kernelList){
    if(kernelList.size() == 1){
      return kernelList.get(0).getReadOnlyOuterIdSet();
    }
    
    Iterator<ACCgpuKernel> kernelIter = kernelList.iterator();
    ACCgpuKernel kernel = kernelIter.next();
    Set<Ident> readOnlyOuterIdSet = kernel.getReadOnlyOuterIdSet();
    while(kernelIter.hasNext()){
      kernel = kernelIter.next();
      readOnlyOuterIdSet.addAll(kernel.getReadOnlyOuterIdSet());
    }
    
    for(ACCgpuKernel kern : kernelList){
      Set<Ident> outerIdSet = new HashSet<Ident>(kern.getOuterIdSet());
      outerIdSet.removeAll(kern.getReadOnlyOuterIdSet());
      readOnlyOuterIdSet.removeAll(outerIdSet);
    }
    
    return readOnlyOuterIdSet;
  }

}



