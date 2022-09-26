/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.util;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import exc.object.XobjectFile;

import exc.openacc.ACC;
import exc.openacc.AccDevice;
import exc.openacc.AccTranslator;

import exc.openmp.OMP;
import exc.openmp.OMPtranslate;
import exc.OMPtoACC.OMPtoACC;
import exc.openmp.OMPDDRD;

import exc.xcodeml.XcodeMLtools;
import exc.xcodeml.XcodeMLtools_F;
import exc.xcodeml.XcodeMLtools_Fmod;
import exc.xcodeml.XcodeMLtools_C;

import xcodeml.util.*;
import exc.xcodeml.XmXobjectToXcodeTranslator;
import exc.xcodeml.XmfXobjectToXcodeTranslator;
import exc.xcodeml.XmcXobjectToXcodeTranslator;
import org.w3c.dom.Document;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

/* only for omp/ompacc */
public class omp2
{
  private static void error(String s)
  {
    System.err.println(s);
    System.exit(1);
  }
    
  private static void usage()
  {
    final String[] lines = {
      "arguments: [-xc|-xf] [-l] [-fopenmp] [-fopenmp-target] [-f[no]coarray] [-dxcode] [-ddecomp] [-dump]",
      "           <input XcodeML file>",
      "           [-o <output reconstructed XcodeML file>]",
      "",
      "  -xc                   process XcodeML/C document.",
      "  -xf                   process XcodeML/Fortran document.",
      "  -l                    suppress line directive in decompiled code.",
      "  -fopenmp              enable OpenMP translation.",
      "  -fopenmp-target       enable OpenMP target translation.",
      "  -facc                 enable OpenACC",
      "  -fopenacc             enable OpenACC",
      "  -w N                  set max columns to N for Fortran source.",
      "  -gnu                  decompile for GNU Fortran (default).",
      "  -intel                decompile for Intel Fortran.",
      "  -M dir                specify where to search for .xmod files",
      "  -decomp               output decompiled source code.",
      "  -silent               no output.",
      "  -rename_main=NAME",
      "                        rename the main function NAME.",
      "  -pointer_size=N,N,N   set pointer sizes (scalar, array, difference between ranks)",
      "",
      " Debug Options:",
      "  -d                    enable output debug message.",
      "  -dxcode               output Xcode file as <input file>.x",
      "  -dump                 output Xcode file and decompiled file to standard output.",
      "  -domp                 enable output OpenMP translation debug message.",
      " Profiling Options:",
      "  -tlog-all             output results in tlog format for all directives.",
      "  -tlog-selective       output results in tlog format for selected directives.",
    };
        
    for(String line : lines) {
      System.err.println(line);
    }
    System.exit(1);
  }
    
  public static void main(String[] args) throws Exception
  {
    String inXmlFile           = null;
    String outXmlFile          = null;
    String lang                = "C";
    boolean openMP             = false;
    boolean openMPTarget       = false;
    boolean openACC            = false;
    boolean async              = false;
    boolean outputXcode        = false;
    boolean outputDecomp       = false;
    boolean dump               = false;
    boolean all_profile        = false;
    boolean selective_profile  = false;
    boolean doTlog             = false;
    boolean silent             = false;
    int maxColumns             = 0;
    int accDefaultVectorLength = 0;
    boolean accDisableReadOnlyDataCache = false;
        
    for(int i = 0; i < args.length; ++i) {
      String arg = args[i];
      String narg = (i < args.length - 1) ? args[i + 1] : null;
    
      if(arg.equals("-h") || arg.equals("--help")) {
        usage();
      } else if(arg.equals("-xc")) {
        lang = "C";
      } else if(arg.equals("-xf")) {
        lang = "F";
      } else if(arg.equals("-l")) {
        XmOption.setIsSuppressLineDirective(true);
      } else if(arg.equals("-fopenmp")) {
        openMP = true;
      } else if(arg.equals("-fopenmp-target")) {
        openMPTarget = true;
      } else if(arg.equals("-facc")) {
        openACC = true; 
      } else if(arg.equals("-fopenacc")) {
        openACC = true; 
      } else if(arg.equals("-fasync")) {
        async = true;
      } else if(arg.equals("-w")) {
        if(narg == null)
          error("needs argument after -w");
        maxColumns = Integer.parseInt(narg);
        ++i;
      } else if(arg.equals("-dxcode")) {
        outputXcode = true;
      } else if(arg.equals("-decomp")) {
        outputDecomp = true;
      } else if(arg.equals("-silent")){
        silent = true;
      } else if(arg.startsWith("-pointer_size=")){
        String   s = arg.substring(arg.indexOf("=") + 1);
        String[] v = s.split(",", 0);
        int pointer_scalar_size = Integer.parseInt(v[0]);
        int pointer_array_size  = Integer.parseInt(v[1]);
        int pointer_diff_size   = Integer.parseInt(v[2]);
        XmOption.setPointerScalarSize(pointer_scalar_size);
        XmOption.setPointerArraySize(pointer_array_size);
        XmOption.setPointerDiffSize(pointer_diff_size);
      } else if(arg.startsWith("-rename_main=")) {
        String main_name = arg.substring(arg.indexOf("=") + 1);
        XmOption.setMainName(main_name);
      } else if(arg.equals("-dump")) {
        dump = true;
        outputXcode = true;
        outputDecomp = true;
      } else if(arg.equals("-d")) {
        XmOption.setDebugOutput(true);
      } else if(arg.equals("-domp")) {
        OMP.debugFlag = true;
      } else if(arg.equals("-o")) {
        if(narg == null)
          error("needs argument after -o");
        outXmlFile = narg;
        ++i;
      } else if(arg.equals("-gnu")) {
        XmOption.setCompilerVendor(XmOption.COMP_VENDOR_GNU);
      } else if(arg.equals("-intel")) {
        XmOption.setCompilerVendor(XmOption.COMP_VENDOR_INTEL);
      } else if (arg.equals("-tlog-selective")) {
        selective_profile = true;
        doTlog = true;
      } else if (arg.equals("-tlog-all")) {
        all_profile = true;
        doTlog = true;
      } else if (arg.startsWith("-M")) { 
          if (arg.equals("-M")) {
            if (narg == null)
              error("needs argument after -M");
            XcodeMLtools_Fmod.addSearchPath(narg);
            ++i;
          } else {
            XcodeMLtools_Fmod.addSearchPath(arg.substring(2));
          }
      } else if (arg.equals("-no-ldg")) {
        accDisableReadOnlyDataCache = true;
      } else if (arg.startsWith("-default-veclen=")) {
        String n = arg.substring("-default-veclen=".length());
        accDefaultVectorLength = Integer.parseInt(n);
      } else if (arg.startsWith("-platform=")){
        String n = arg.substring("-platform=".length());
        ACC.platform = ACC.Platform.valueOf(n);
      } else if (arg.startsWith("-device=")){
        String n = arg.substring("-device=".length());
        ACC.device = AccDevice.getDevice(n);
      } else if(inXmlFile == null) {
        inXmlFile = arg;
      } else {
        error("too many arguments");
      }
    }
        
    Reader reader = null;
    File dir      = null;
    if(inXmlFile == null) {
      reader = new InputStreamReader(System.in);
    }
    else {
      reader = new BufferedReader(new FileReader(inXmlFile));
      dir = new File(inXmlFile).getParentFile();
    }

    Writer xcodeWriter = null;
    if(dump || outputXcode) {
      if(dump) {
        xcodeWriter = new OutputStreamWriter(System.out);
      } else {
        xcodeWriter = new BufferedWriter(new FileWriter(inXmlFile + ".x"));
      }
    }
   
    XmOption.setLanguage(XmLanguage.valueOf(lang));
    XmOption.setIsOpenMP(openMP);
    XmOption.setIsOpenMPTarget(openMPTarget);
    XmOption.setIsAsync(async);
    XmOption.setTlogMPIisEnable(doTlog);
    
    // read XcodeML
    XcodeMLtools tools = 
      (XmOption.getLanguage() == XmLanguage.F)? 
      new XcodeMLtools_F() : new XcodeMLtools_C();
    XobjectFile xobjFile = tools.read(reader);
    
    if (inXmlFile != null) reader.close();
    if (xobjFile == null)  System.exit(1);

    String srcPath  = inXmlFile;
    String baseName = null;

    if(dump || srcPath == null || srcPath.indexOf("<") >= 0 ) {
      srcPath = null;
    }
    else {
      String fileName = new File(srcPath).getName();
      int idx = fileName.lastIndexOf(".");
      if(idx < 0) {
        XmLog.fatal("invalid source file name : " + fileName);
      }
      baseName = fileName.substring(0, idx);
    }

    // Output Xcode
    if(xcodeWriter != null) {
      xobjFile.Output(xcodeWriter);
      xcodeWriter.flush();
    }
        
    System.gc();
        
    if (outputXcode) {
      Writer dumpWriter = new BufferedWriter(new FileWriter(inXmlFile +
                                                            ".xobj.1.dump"));
      xobjFile.Output(dumpWriter);
      dumpWriter.close();
    }

    // OpenMP translation
    if(openMPTarget){
      // xobjFile.addHeaderLine("#include \"ompc_target.h\""); /* ???? */
      System.out.println("OpenMP target (OMPtoACC) ...");
        
      OMPtoACC ompToAccTranslator = new OMPtoACC(xobjFile);
      xobjFile.iterateDef(ompToAccTranslator);
      
      if(OMP.hasErrors())
        System.exit(1);
      
      ompToAccTranslator.finish();
      openACC = true; // enable OpenACC pass
    } else
      if(openMP) {
        OMPtranslate ompTranslator = new OMPtranslate(xobjFile);
        xobjFile.iterateDef(ompTranslator);
        
        if(OMP.hasErrors())
          System.exit(1);
            
        ompTranslator.finish();
        
        if(xcodeWriter != null) {
          xobjFile.Output(xcodeWriter);
          xcodeWriter.flush();
        }
      } 

    if(openACC){
      System.out.println("OpenACC ...");
      if(ACC.device == AccDevice.NONE){
        switch(ACC.platform){
        case CUDA:
        case OpenCL:
          ACC.device = AccDevice.getDevice("Fermi"); // default?
          break;
        }
      }
      ACC.init();

      if(accDefaultVectorLength > 0) {
        ACC.device.setDefaultVectorLength(accDefaultVectorLength);
      }

      if(accDisableReadOnlyDataCache == true){
        ACC.device.setUseReadOnlyDataCache(false);
      }

      //XmOption.setDebugOutput(true);
      AccTranslator accTranslator = new AccTranslator(xobjFile, false);
      xobjFile.iterateDef(accTranslator);

      accTranslator.finish();
      
      if(xcodeWriter != null) {
        xobjFile.Output(xcodeWriter);
        xcodeWriter.flush();
      }
    } /* OpenACC */

    if(!dump && outputXcode) {
      xcodeWriter.close();
    }

    // translate Xcode to XcodeML
    // create transformer from Xobject to XcodeML DOM.
    XmXobjectToXcodeTranslator xc2xcodeTranslator = null;
    if (lang.equals("F"))
      xc2xcodeTranslator = new XmfXobjectToXcodeTranslator();
    else
      xc2xcodeTranslator = new XmcXobjectToXcodeTranslator();
    
    Document xcodeDoc = xc2xcodeTranslator.write(xobjFile);

    // transformation from DOM to the file. It means to output DOM to the file.
    // System.out.println("silent="+silent);
    // silent = false;
    if(silent == false){
      try {
        Transformer transformer = TransformerFactory.newInstance().newTransformer();
        transformer.setOutputProperty(OutputKeys.METHOD, "xml");
        Writer xmlWriter = null;
        if (outXmlFile == null)
          xmlWriter = new OutputStreamWriter(System.out);
        else
          xmlWriter = new BufferedWriter(new FileWriter(outXmlFile));
        
        transformer.transform(new DOMSource(xcodeDoc), new StreamResult(xmlWriter));
        xmlWriter.flush();
        
        if(outXmlFile != null) {
          xmlWriter.close();
          xmlWriter = null;
        }
      } catch(TransformerException e) {
        throw new XmException(e);
      }
    }
    
    // Decompile
    XmToolFactory toolFactory = new XmToolFactory(lang);
    XmDecompilerContext context = toolFactory.createDecompilerContext();
    if(lang.equals("F")) {
      if(maxColumns > 0)
        context.setProperty(XmDecompilerContext.KEY_MAX_COLUMNS, "" + maxColumns);
    }
        
    if(outputDecomp) {
      Writer decompWriter = null;
      if(dump || srcPath == null) {
        decompWriter = new OutputStreamWriter(System.out);
      } 
      else { // set decompile writer
        String newFileName = baseName + "." + (XmOption.isLanguageC() ? "c" : "F90");
        File newFile = new File(dir, newFileName);
                
        if(newFile.exists())
          newFile.renameTo(new File(dir, newFileName + ".i"));
                
        decompWriter = new BufferedWriter(new FileWriter(newFile));
      }

      if (xcodeDoc == null) {
        javax.xml.parsers.DocumentBuilderFactory docFactory = 
          javax.xml.parsers.DocumentBuilderFactory.newInstance();
        javax.xml.parsers.DocumentBuilder builder = docFactory.newDocumentBuilder();
        xcodeDoc = builder.parse(outXmlFile);
      }
      
      XmDecompiler decompiler = toolFactory.createDecompiler();
      decompiler.decompile(context, xcodeDoc, decompWriter);
      decompWriter.flush();
    
      if(!dump && outputDecomp) {
        decompWriter.close();
      }
    }
  }
}
