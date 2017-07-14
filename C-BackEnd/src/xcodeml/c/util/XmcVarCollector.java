package xcodeml.c.util;

import org.w3c.dom.*;
import java.util.Set;

/**
 * Collector of variable references.
 */
public class XmcVarCollector {
    public XmcVarCollector(Set<String> variables) {
        this.variables = variables;
    }

    public void collectVars(Node n){
        String nodeName = n.getNodeName();      
        Boolean isVarNode = nodeName.equals("Var")
                || nodeName.equals("varAddr")
                || nodeName.equals("funcAddr")
                || nodeName.equals("memberRef");

        NodeList cnList = n.getChildNodes();
        for(int i=0; i<cnList.getLength(); i++){
            Node cn = cnList.item(i);
            if(isVarNode && cn.getNodeType() == Node.TEXT_NODE){
                variables.add(cn.getNodeValue());
            }else{
                collectVars(cn);
            }
        }
    }

    private Set<String> variables;
}
