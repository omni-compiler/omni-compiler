package exc.openacc;

import java.util.*;

import exc.block.*;
import exc.object.*;


public class ACCtranslateParallel {
  private PragmaBlock pb;
  private ACCinfo parallelInfo;
  private ACCglobalDecl globalDecl;
 
  public ACCtranslateParallel(PragmaBlock pb) {
    this.pb = pb;
    this.parallelInfo = ACCutil.getACCinfo(pb);
    if(this.parallelInfo == null){
      ACC.fatal("can't get info");
    }
    this.globalDecl = this.parallelInfo.getGlobalDecl();
  }
  
  public void translate() throws ACCexception{
    ACC.debug("translate parallel");
    
    if(parallelInfo.isDisabled()){
      return;
    }
    
    List<Block> kernelBody = new ArrayList<Block>(); 
    if(parallelInfo.getPragma() == ACCpragma.PARALLEL_LOOP){
      kernelBody.add(pb);
    }else{
//      for(Block b = pb.getBody().getHead(); b != null; b = b.getNext()){
//        kernelBody.add(b);
//      }
        kernelBody.add(pb);
    }
    
    //analyze and complete clause for kernel
    
    ACCgpuKernel gpuKernel = new ACCgpuKernel(parallelInfo, kernelBody);
    gpuKernel.analyze();
    
    //get readonly id set
    Set<Ident> readOnlyOuterIdSet = gpuKernel.getReadOnlyOuterIdSet();//collectReadOnlyOuterIdSet(kernelList);
    
    //kernel内のincudtionVarのid
    //Set<Ident> inductionVarIdSet = gpuKernel.getInductionVarIdSet();

    //set unspecified var's attribute from outerIdSet
    Set<Ident> outerIdSet = new HashSet<Ident>(gpuKernel.getOuterIdList());
    for(Ident id : outerIdSet){
      String varName = id.getName();
      //if(parallelInfo.getACCvar(varName) != null) continue; 
      if(parallelInfo.isVarAllocated(varName)) continue;
      ACCvar var = parallelInfo.getACCvar(varName);
      if(var != null && var.isAllocated()) continue;
      //if(parallelInfo.isVarPrivate(varName)) continue;
      if(parallelInfo.isVarFirstprivate(varName)) continue; //this is need for only parallel construct
      //if(parallelInfo.getDevicePtr(varName) != null) continue;
      if(parallelInfo.isVarReduction(varName)) continue;
      
      if(readOnlyOuterIdSet.contains(id) && !id.Type().isArray()) continue; //firstprivateは除く
      //if(inductionVarIdSet.contains(id)) continue;
      
      parallelInfo.declACCvar(id.getName(), ACCpragma.PRESENT_OR_COPY);
    }
    
    //translate data
    ACCtranslateData dataTranslator = new ACCtranslateData(pb);
    dataTranslator.translate();
    
    //make kernels list of block(kernel call , sync) 
    
    Block parallelBlock = gpuKernel.makeLaunchFuncCallBlock();
    Block replaceBlock = null;
    if(parallelInfo.isEnabled()){
      replaceBlock = parallelBlock;
    }else{
      replaceBlock = Bcons.IF(parallelInfo.getIfCond(), parallelBlock, Bcons.COMPOUND(pb.getBody()));
    }

    //set replace block
    parallelInfo.setReplaceBlock(replaceBlock);
  }
}

  