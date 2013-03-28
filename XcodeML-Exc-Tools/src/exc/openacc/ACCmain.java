package exc.openacc;
import java.io.*;
import java.util.*;

import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;

import com.sun.org.apache.xml.internal.serializer.OutputPropertiesFactory;

import exc.object.XobjectFile;
import exc.xcodeml.*;

import xcodeml.*;
import xcodeml.binding.XmXcodeProgram;
import xcodeml.util.*;

public class ACCmain
{

  static boolean dump_flag = true;

  private static void error(String s)
  {
    System.err.println(s);
    System.exit(1);
  }

  public static void main(String[] args) throws Exception
  {
    String inXmlFile = null;
    String outXmlFile = null;
    String outFile = null;
    String lang = "C";
    boolean dump = false;
    boolean indent = true; //
    boolean outputDecomp = true;

    System.out.println("test_main_C ... #args="+args.length);
    inXmlFile = args[0];
    outXmlFile = "out.xml";
    outFile = "test.c";
    System.out.println("input file = "+inXmlFile);

    Reader reader = null;
    Writer xmlWriter = null;
    Writer xcodeWriter = null;
    Writer decompWriter = null;
    File dir = null;

    if(inXmlFile == null) {
      error("no input file");
    } else {
      reader = new BufferedReader(new FileReader(inXmlFile));
      dir = new File(inXmlFile).getParentFile();
    }

    if(outXmlFile == null) {
      error("no output file");
    } else {
      xmlWriter = new BufferedWriter(new FileWriter(outXmlFile));
    }

    if(outFile == null) {
      error("no output file");
    } else {
      decompWriter = new BufferedWriter(new FileWriter(outFile));
    }

    
//    if(dump || outputXcode) {
//      if(dump) {
//        xcodeWriter = new OutputStreamWriter(System.out);
//      } else {
//        xcodeWriter = new BufferedWriter(new FileWriter(inXmlFile + ".x"));
//      }
//    }
    
    //xcodeWriter = new OutputStreamWriter(System.out);

    XmToolFactory toolFactory = new XmToolFactory(lang);

    XmOption.setLanguage(XmLanguage.valueOf(lang));
    XmOption.setDebugOutput(true);

    XcodeMLtools tools = null;
    if (XmOption.getLanguage() == XmLanguage.F) {
      tools = new XcodeMLtools_F();
    } else {
      tools = new XcodeMLtools_C();
    }
    // read XcodeML
    XobjectFile xobjFile = tools.read(reader);
    if (inXmlFile != null) {
      reader.close();
    }

    if(xobjFile == null)
      System.exit(1);

    // Output Xcode
    if(dump_flag && xcodeWriter != null) {
      System.out.println("*** dump Xobject (after read) ...");
      xobjFile.Output(xcodeWriter);
      xcodeWriter.flush();
    }

    System.gc();

    // translate OpenACC
    {
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

      System.out.println("*** rewrite xml ...");
      ACCrewritePragma accRewriter = new ACCrewritePragma(accGlobalDecl);
      xobjFile.iterateDef(accRewriter);
      accRewriter.finalize();
      ACC.exitByError();

      accGlobalDecl.setupGlobalConstructor();
      accGlobalDecl.setupGlobalDestructor();
      ACC.exitByError();

      xobjFile.addHeaderLine("include \"acc.h\"");
      xobjFile.addHeaderLine("include \"acc_gpu.h\"");
      accGlobalDecl.finalize();

    }


    if(dump_flag && xcodeWriter != null) {
      System.out.println("*** dump Xobject (after ACC translate) ...");
      xobjFile.Output(xcodeWriter);
      xcodeWriter.flush();
    }

    
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
        /*
          String newFileName = baseName + "." + (XmOption.isLanguageC() ? "c" : "F90");
          String newFileName2 = baseName + "." + (XmOption.isLanguageC() ? "c" : "f90");
          File newFile = new File(dir, newFileName);
          File newFile2 = new File(dir, newFileName2);

          if(newFile.exists())
            newFile.renameTo(new File(dir, newFileName + ".i"));
          if(newFile2.exists())
            newFile2.renameTo(new File(dir, newFileName2 + ".i"));

          decompWriter = new BufferedWriter(new FileWriter(newFile));
         */
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

  }
}
