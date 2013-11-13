package exc.openacc;
import java.io.*;
import java.util.*;

import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;

import com.sun.org.apache.xml.internal.serializer.OutputPropertiesFactory;

import exc.block.FuncDefBlock;
import exc.object.FunctionType;
import exc.object.Ident;
import exc.object.Xobject;
import exc.object.XobjectFile;
import exc.object.Xtype;
import exc.openmp.OMP;
import exc.xcalablemp.XMPexception;
import exc.xcodeml.*;
import xcodeml.*;
import xcodeml.binding.XmXcodeProgram;
import xcodeml.util.*;

public class ACCmain
{

  //static boolean dump_flag = true;

  private static void error(String s)
  {
    System.err.println(s);
    System.exit(1);
  }
  
  private static void usage() {
    // TODO Auto-generated method stub
    
  }

  public static void main(String[] args) throws Exception
  {
    String inXmlFile = null;
    String outXmlFile = null;
    String decompName = null;
    String lang = "C";
    boolean outputXcode = false;
    boolean outputDecomp = false;//true;
    boolean dump = false;
    boolean indent = true; //false
        
    for(int i = 0; i< args.length; ++i){
      String arg = args[i];
      String narg = (i < args.length - 1)? args[i + 1] : null;
      
      if(arg.equals("-h") || arg.equals("--help")){
        usage();
        return;
      } else if(arg.equals("-xc")) {
        lang = "C";
      } else if(arg.equals("-xf")) {
        lang = "F";
      } else if(arg.equals("-l")) {
        XmOption.setIsSuppressLineDirective(true);
      } else if(arg.equals("-i")) {
        indent = true;
        //      } else if(arg.equals("-fopenmp")) {
        //  openMP = true;
        //      } else if(arg.equals("-fxmp")) {
        //  xcalableMP = true;
        //      } else if(arg.equals("-w")) {
        //        if(narg == null)
        //          error("needs argument after -w");
        //        maxColumns = Integer.parseInt(narg);
        //        ++i;
      } else if(arg.equals("-dxcode")) {
        outputXcode = true;
      } else if(arg.equals("-decomp")) {
        outputDecomp = true;
      } else if(arg.equals("-dump")) {
        dump = true;
        indent = true;
        outputXcode = true;
        outputDecomp = true;
      } else if(arg.equals("-d")) {
        XmOption.setDebugOutput(true);
//      } else if(arg.equals("-fatomicio")) {
//        XmOption.setIsAtomicIO(true);
      } else if(arg.equals("-domp")) {
        OMP.debugFlag = true;
      } else if(arg.equals("-dxmp")) {
        exc.xmpF.XMP.debugFlag = true;
      } else if(arg.equals("-o")) {
        if(narg == null)
          error("needs argument after -o");
        outXmlFile = narg;
        ++i;
      } else if(arg.startsWith("-")){
        error("unknown option " + arg);
      } else if(inXmlFile == null) {
        inXmlFile = arg;
      } else {
        error("too many arguments");
      }
    }

    //System.out.println("test_main_C ... #args="+args.length);
    //inXmlFile = args[0];
    //outXmlFile = "out.xml";
    //decompName = "test.c";
    //System.out.println("input file = "+inXmlFile);

    Reader reader = null;
    Writer xmlWriter = null;
    Writer xcodeWriter = null;
    Writer decompWriter = null;
    File dir = null;

    if(inXmlFile == null) {
      error("no input file");
      //reader = new InputStreamReader(System.in);
    } else {
      reader = new BufferedReader(new FileReader(inXmlFile));
      dir = new File(inXmlFile).getParentFile();
    }

    if(outXmlFile == null) {
      //error("no output file");
      xmlWriter = new OutputStreamWriter(System.out);
    } else {
      xmlWriter = new BufferedWriter(new FileWriter(outXmlFile));
    }

    if(decompName == null) {
      //error("no output file");
    } else {
      decompWriter = new BufferedWriter(new FileWriter(decompName));
    }

    
    if(dump || outputXcode) {
      if(dump) {
        xcodeWriter = new OutputStreamWriter(System.out);
      } else {
        xcodeWriter = new BufferedWriter(new FileWriter(inXmlFile + ".x"));
      }
    }
    
    XmToolFactory toolFactory = new XmToolFactory(lang);
    XmOption.setLanguage(XmLanguage.valueOf(lang));

    XcodeMLtools tools = null;
    if (XmOption.getLanguage() == XmLanguage.C) {
      tools = new XcodeMLtools_C();
    } else {
      tools = new XcodeMLtools_F();
    }
    // read XcodeML
    XobjectFile xobjFile = tools.read(reader);
    if (inXmlFile != null) {
      reader.close();
    }

    if(xobjFile == null){
      error("read XcodeML error");
    }
    
    // Output Xcode
    if(xcodeWriter != null) {
      System.out.println("*** dump Xobject (before read) ...");
      xobjFile.Output(xcodeWriter);
      xcodeWriter.flush();
    }
    
 

    System.gc();

    // translate OpenACC
    if(false){
      System.out.println("*** analyze ACC ...");
      ACCglobalDecl accGlobalDecl = new ACCglobalDecl(xobjFile);
      ACCanalyzePragma accAnalyzer = new ACCanalyzePragma(accGlobalDecl);
      xobjFile.iterateDef(accAnalyzer);
      accAnalyzer.finalize();
      ACC.exitByError();

      System.out.println("*** translate ACC ...");
      ACCtranslatePragma accTranslator = new ACCtranslatePragma(accGlobalDecl);
      xobjFile.iterateDef(accTranslator);
      accTranslator.finalize();
      ACC.exitByError();

      System.out.println("*** rewrite code ...");
      ACCrewritePragma accRewriter = new ACCrewritePragma(accGlobalDecl);
      xobjFile.iterateDef(accRewriter);
      accRewriter.finalize();
      ACC.exitByError();

      accGlobalDecl.setupGlobalConstructor();
      accGlobalDecl.setupGlobalDestructor();
      accGlobalDecl.setupMain();
      ACC.exitByError();

      xobjFile.addHeaderLine("include \"acc.h\"");
      xobjFile.addHeaderLine("include \"acc_gpu.h\"");
      accGlobalDecl.finalize();

    }


    if(xcodeWriter != null) {
      System.out.println("*** dump Xobject (after ACC translate) ...");
      xobjFile.Output(xcodeWriter);
      xcodeWriter.flush();
    }

    // translate Xcode to XcodeML
    System.out.println("*** translate Xcode to XcodeML ...");
    XmXcodeProgram xmprog = null;
    Document xcodeDoc = null;

    XmXobjectToXcodeTranslator xc2xcodeTranslator = null;
    if (lang.equals("F")) {
      xc2xcodeTranslator = new XmfXobjectToXcodeTranslator();
    } else {
      xc2xcodeTranslator = new XmcXobjectToXcodeTranslator();
    }
    xcodeDoc = xc2xcodeTranslator.write(xobjFile);

    Transformer transformer = null;
    try {
      transformer = TransformerFactory.newInstance().newTransformer();
    } catch(TransformerConfigurationException e) {
      throw new XmException(e);
    }

    transformer.setOutputProperty(OutputKeys.METHOD, "xml");

    if (indent) {
      final int indentSpaces = 2;
      transformer.setOutputProperty(OutputKeys.INDENT, "yes");
      transformer.setOutputProperty(OutputPropertiesFactory.S_KEY_INDENT_AMOUNT, "" + indentSpaces);
    }
    try {
      transformer.transform(new DOMSource(xcodeDoc), new StreamResult(xmlWriter));
    } catch(TransformerException e) {
      throw new XmException(e);
    }

    if (!dump && !outputDecomp) {
      xmprog = null;
    } else {
      // read XcodeML/C again. Make xmprog.
      if (outXmlFile != null && false) {
        reader = new BufferedReader(new FileReader(outXmlFile));
        List<String> readErrorList = new ArrayList<String>();
        xmprog = toolFactory.createXcodeProgram();
        XmValidator validator = toolFactory.createValidator();
        if (!validator.read(reader, xmprog, readErrorList)) {
          for (String error : readErrorList) {
            System.err.println(error);
            System.exit(1);
          }
        }
        reader.close();
        xcodeDoc = null;
      }
    }

    xmlWriter.flush();

    if(outXmlFile != null) {
      xmlWriter.close();
      xmlWriter = null;
    }

    System.out.println("*** Decompile XcodeML ...");

    XmDecompilerContext context = null;

    if(outputDecomp) {
      if(dump /*|| srcPath == null*/) {
        decompWriter = new OutputStreamWriter(System.out);
      } else {
        // set decompile writer
    	  if(decompName == null){
    	  String srcName = xobjFile.getSourceFileName();
    	  String baseName = ACCutil.removeExtension(srcName);
    	  decompName = baseName + "." + "i";
    	  }

    	  File decompFile = new File(dir, decompName);
    	  //File newFile2 = new File(dir, newFileName2);
//
//    	  if(newFile.exists())
//    		  newFile.renameTo(new File(dir, newFileName + ".i"));
//    	  if(newFile2.exists())
//    		  newFile2.renameTo(new File(dir, newFileName2 + ".i"));

    	  decompWriter = new BufferedWriter(new FileWriter(decompFile));

      }

      XmDecompiler decompiler = toolFactory.createDecompiler();
      if (xcodeDoc == null) {
        javax.xml.parsers.DocumentBuilderFactory docFactory = javax.xml.parsers.DocumentBuilderFactory.newInstance();
        javax.xml.parsers.DocumentBuilder builder = docFactory.newDocumentBuilder();
        xcodeDoc = builder.parse(outXmlFile);
      }
      decompiler.decompile(context, xcodeDoc, decompWriter);

      decompWriter.flush();

      if(!dump && outputDecomp) {
        decompWriter.close();
      }
    }
    System.out.println("*** End");
  }
}
