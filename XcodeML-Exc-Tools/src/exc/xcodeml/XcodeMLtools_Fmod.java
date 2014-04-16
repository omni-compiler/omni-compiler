package exc.xcodeml;

import java.util.Vector;
import java.io.*;

import org.w3c.dom.*;
import org.xml.sax.InputSource;

import javax.xml.parsers.*;
import javax.xml.transform.dom.*;
import exc.object.*;

import static xcodeml.util.XmDomUtil.getAttr;
import static xcodeml.util.XmDomUtil.getAttrBool;
import static xcodeml.util.XmDomUtil.getContent;
import static xcodeml.util.XmDomUtil.getContentText;
import static xcodeml.util.XmDomUtil.getElement;

/**
 * tools for XcodeML/Fortran Module file to Xcode translation.
 */
public class XcodeMLtools_Fmod extends XcodeMLtools_F {

  String module_name;
  Vector<Xobject> aux_info;

  static private Vector<String> search_path;

  static {
    search_path = new Vector<String>();
  }

  // constructor
  public XcodeMLtools_Fmod() { }

  public String getModuleName() { return module_name; }

  public Vector<Xobject> getAuxInfo() { return aux_info; }

  public XobjectFile read(Reader reader) {
    Document doc = readDocument(reader);
    Node rootNode = doc.getDocumentElement();
    if (rootNode.getNodeName() != "OmniFortranModule")
      fatal("Toplevel must be OmniFortranModule");

    xobjFile = new XobjectFile();

    Node n, nn;
    NodeList list;

    n = getElement(rootNode, "name");
    module_name = n.getFirstChild().getNodeValue();

    aux_info = new Vector<Xobject>();

    // process depends
    n = getElement(rootNode, "depends");
    list = n.getChildNodes();
    for (int i = 0; i < list.getLength(); i++) {
      nn = list.item(i);
      if (nn.getNodeType() != Node.ELEMENT_NODE) continue;

      String module_name = nn.getFirstChild().getNodeValue();

      String mod_file_name = module_name + ".xmod";
      Reader reader0 = null;

      String mod_file_name_with_path = "";
      boolean found = false;
      File mod_file;

      for (String spath: search_path){
	mod_file_name_with_path = spath + "/" + mod_file_name;
	mod_file = new File(mod_file_name_with_path);
	if (mod_file.exists()){
	  found = true;
	  break;
	}
      }

      if (!found){
	mod_file_name_with_path = mod_file_name;
	mod_file = new File(mod_file_name_with_path);
	if (mod_file.exists()){
	  found = true;
	}
      }

      if (!found){
	fatal("module file '"+mod_file_name+"' not found");
      }

      try {
	reader0 = new BufferedReader(new FileReader(mod_file_name_with_path));
      }
      catch(Exception e){
	fatal("cannot open module file '" + mod_file_name + "'");
      }
      XcodeMLtools_Fmod tools = new XcodeMLtools_Fmod();
      tools.read(reader0);

      aux_info.addAll(tools.getAuxInfo());
    }

    // process type table part
    n = getElement(rootNode, "typeTable");
    list = n.getChildNodes();
    for (int i = 0; i < list.getLength(); i++) {
      nn = list.item(i);
      if (nn.getNodeType() != Node.ELEMENT_NODE)
	continue;
      enterType(nn);
    }

    // fix type
    xobjFile.fixupTypeRef();

    // process identifier
    n = getElement(rootNode, "identifiers");
    xobjFile.setIdentList(toIdentList(n));

    // process interface decl 
    // n = getElement(rootNode, "interfaceDecls");

    n = getElement(rootNode, "aux_info");
    //aux_info = new Vector<Xobject>();
    list = n.getChildNodes();
    for (int i = 0; i < list.getLength(); i++) {
      nn = list.item(i);
      if (nn.getNodeType() != Node.ELEMENT_NODE)
	continue;
      aux_info.add(toXobject(nn));
    }

    setIdentDecl(xobjFile);

    return xobjFile;
  }


  static public void addSearchPath(String path){
    search_path.addElement(path);
  }

  static public Vector<String> getSearchPath(){
    return search_path;
  }

}
